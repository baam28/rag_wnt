"""Tool: text-to-sql agent for database ERP query."""

import logging
import re
import threading
from typing import Any, Tuple

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent

from config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLDatabase singleton cache
# ---------------------------------------------------------------------------
# SQLDatabase.from_uri() creates a SQLAlchemy engine and introspects the schema.
# Both are expensive: engine creation opens DB connections outside the app pool,
# and schema introspection runs information_schema queries. Cache per table-set
# so every query reuses the same engine and its built-in connection pool.
# The LLM (toolkit + agent) is still built per-call because it is runtime-
# swappable from the admin panel without restarting the server.

_sql_db_lock = threading.Lock()
_sql_db_cache: dict[frozenset, SQLDatabase] = {}


def _get_sql_db(tables: list[str]) -> SQLDatabase:
    key = frozenset(tables)
    with _sql_db_lock:
        if key not in _sql_db_cache:
            settings = get_settings()
            db_url = settings.database_url
            if db_url.startswith("postgresql://") and "psycopg" not in db_url:
                db_url = db_url.replace("postgresql://", "postgresql+psycopg://")
            _sql_db_cache[key] = SQLDatabase.from_uri(db_url, include_tables=tables)
        return _sql_db_cache[key]


# ---------------------------------------------------------------------------
# Price-query detection (used by the supervisor / router to auto-route)
# ---------------------------------------------------------------------------

_PRICE_KEYWORDS = [
    "giá thuốc", "giá bán", "giá của", "bao nhiêu tiền",
    "thuốc .+ giá", "giá .+ bao nhiêu", "mua .+ bao nhiêu",
    "tra cứu giá", "price of", "có bao nhiêu thuốc", "liệt kê",
    "thuốc nào", "tồn kho", "số lượng tồn kho",
    r"giá\s+.+",
    r".+\s+giá\s*",
    r"bao nhiêu\s+.+",
    r".+\s+bao nhiêu",
]
_PRICE_RE = re.compile("|".join(_PRICE_KEYWORDS), re.IGNORECASE)

def detect_price_query(question: str) -> Tuple[bool, str]:
    """Return (is_price_query, extracted_drug_name)."""
    q = question.strip()
    if not _PRICE_RE.search(q):
        return False, ""
    return True, q

# ---------------------------------------------------------------------------
# Text to SQL Agent Wrapper
# ---------------------------------------------------------------------------

def execute_drug_info_query(question: str) -> str:
    """Query drug_list for structured clinical info (active ingredient, dosage form, therapeutic class, etc.)."""
    try:
        db = _get_sql_db(["drug_list"])
    except Exception as e:
        logger.error(f"Failed to get SQLDatabase for drug info: {e}")
        return ""

    from prompts import build_runtime_chat_client
    llm = build_runtime_chat_client(temperature=0.0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        agent_type="tool-calling",
        prefix=(
            "Bạn là trợ lý tra cứu thông tin thuốc từ cơ sở dữ liệu nội bộ. "
            "Bảng `drug_list` chứa các trường: drug_name (tên thuốc), active_ingredient (hoạt chất), "
            "dosage (hàm lượng), dosage_form (dạng bào chế), prescription_class (phân loại đơn kê toa), "
            "therapeutic_class (nhóm trị liệu/phân loại điều trị). "
            "Hãy viết truy vấn PostgreSQL an toàn (CHỈ SELECT) để tìm các thuốc liên quan đến câu hỏi. "
            "Dùng ILIKE để tìm kiếm tương đối theo tên thuốc hoặc hoạt chất. "
            "Chỉ trả về thông tin có trong bảng, không suy diễn thêm. "
            "Nếu không tìm thấy thuốc liên quan, trả về chuỗi rỗng."
        )
    )

    try:
        result = agent_executor.invoke({"input": question})
        output = result.get("output", "")
        return output if isinstance(output, str) else str(output)
    except Exception as e:
        logger.error(f"Drug info query failed for '{question}': {e}", exc_info=True)
        return ""


def execute_erp_query(question: str) -> str:
    """Uses a generative Text-to-SQL agent to answer any ERP/inventory question (price, stock, expiry, packaging)."""
    try:
        db = _get_sql_db(["drug_list", "drug_inventory"])
    except Exception as e:
        logger.error(f"Failed to get SQLDatabase for ERP: {e}")
        return "Xin lỗi, không thể kết nối tới cơ sở dữ liệu để thực hiện truy vấn."

    from prompts import build_runtime_chat_client
    llm = build_runtime_chat_client(temperature=0.0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        agent_type="tool-calling",
        prefix=(
            "Bạn là trợ lý ảo phân tích trực tiếp cơ sở dữ liệu thuốc/kho hàng ERP của hệ thống y tế. "
            "Hai bảng liên kết qua khoá `drug_id`:\n"
            "  - `drug_list`: drug_id (PK), drug_name (tên thuốc), active_ingredient (hoạt chất), "
            "dosage (hàm lượng), dosage_form (dạng bào chế), prescription_class (phân loại kê đơn), "
            "therapeutic_class (nhóm điều trị).\n"
            "  - `drug_inventory`: drug_id (FK), batch_number (mã lô), selling_unit (đơn vị bán: viên/hộp/chai...), "
            "packaging_size (quy cách đóng gói, ví dụ '100 viên/hộp'), "
            "retail_price (giá bán lẻ theo MỘT đơn vị bán, kiểu NUMERIC), "
            "stock_quantity (số lượng tồn kho, kiểu INTEGER), "
            "batch_date (ngày nhập/lô hàng, kiểu DATE), "
            "expiry_date (hạn sử dụng, kiểu DATE).\n"
            "Viết và chạy truy vấn PostgreSQL an toàn (CHỈ SELECT) để trả lời câu hỏi của người dùng. "
            "CHỈ trả về các trường thông tin mà câu hỏi yêu cầu — không liệt kê thêm thành phần, dạng bào chế hay giá nếu người dùng không hỏi. "
            "Ví dụ: nếu hỏi về tồn kho thì chỉ trả về tên thuốc và stock_quantity; nếu hỏi về giá thì chỉ trả về tên thuốc, retail_price và selling_unit. "
            "QUAN TRỌNG về giá: retail_price là giá trên MỘT đơn vị bán (selling_unit). "
            "Khi trình bày giá, LUÔN kèm đơn vị bán — ví dụ '240 VNĐ/viên' hoặc '60.000 VNĐ/hộp'. "
            "KHÔNG bao giờ trình bày giá mà thiếu đơn vị, vì sẽ gây nhầm lẫn giữa giá/viên và giá/hộp. "
            "Trình bày kết quả bằng Markdown Table hoặc danh sách ngắn gọn. "
            "Tự chủ động search tương đối (ILIKE) để bao phủ sai chính tả từ người dùng."
        )
    )
    
    try:
        result = agent_executor.invoke({"input": question})
        return result.get("output", "Không tìm thấy kết quả rõ ràng trong cơ sở dữ liệu.")
    except Exception as e:
        logger.error(f"SQL Tool failed for query '{question}': {e}", exc_info=True)
        return "Xin lỗi, đã có lỗi xảy ra khi truy xuất cơ sở dữ liệu ERP."
