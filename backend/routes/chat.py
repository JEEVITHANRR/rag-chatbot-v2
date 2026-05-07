from flask import Blueprint, request, jsonify
from services.rag_service import query_documents

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        result = query_documents(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
