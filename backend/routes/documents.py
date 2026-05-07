import os
import tempfile
from flask import Blueprint, request, jsonify
from services.rag_service import process_document, get_uploaded_docs

documents_bp = Blueprint("documents", __name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@documents_bp.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only PDF, TXT, and MD files are supported"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        chunks = process_document(tmp_path, file.filename)
        os.unlink(tmp_path)

        return jsonify({
            "message": f"Successfully processed {file.filename}",
            "chunks": chunks,
            "documents": get_uploaded_docs(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@documents_bp.route("/documents", methods=["GET"])
def list_documents():
    return jsonify({"documents": get_uploaded_docs()})
