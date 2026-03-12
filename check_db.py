from pymongo import MongoClient
import json
from bson import ObjectId

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

client = MongoClient('mongodb://localhost:27017/')
db = client['rag_chatbot']

sessions = list(db.chat_sessions.find())
print("Sessions:", json.dumps(sessions, cls=JSONEncoder, indent=2))

messages = list(db.chat_messages.find())
print("Messages:", len(messages))
if len(messages) > 0:
    print("Sample msg:", json.dumps(messages[0], cls=JSONEncoder, indent=2))

users = list(db.users.find())
print("Users:", json.dumps(users, cls=JSONEncoder, indent=2))
