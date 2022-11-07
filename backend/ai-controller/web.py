
import os
import logging
logging.basicConfig(level=os.environ.get("LOGLEVEL"))

import requests
from sklearn.metrics.pairwise import cosine_similarity

from flask import request as flask_req


def _clean_text(text:list) -> list:
	return NotImplementedError()

def _build_request(texts:list) -> dict:
	return {"texts": [{"text_id": i, "text": text} for i, text in enumerate(texts)]}

class EmbedHandler:
	"""
	Class that implements the request handeling for the embedding service
	"""

	def __init__(self, host:str="localhost", port:int=8081, model:str="EmbedBert", version:str="1.0"):
		"""
		Init that creates the embedding service url
		:param host: host ip/ address
		:param port: host port
		:param model: model name
		:param version: version of model to use
		"""
		self.embed_service_url = f"http://{host}:{port}/predictions/{model}/{version}"
		print(self.embed_service_url)

	def predict(self, texts: list) -> dict:
		"""
		Function that returns the embeddings for the given input texts.
		:param texts: list of texts
		:return: list of embeddings
		"""

		# Here you could add potential cleanup code.
		# texts = _clean_text()

		# Making the request
		response_obj = requests.post(self.embed_service_url, json=_build_request(texts))

		return response_obj


@app.route("/", methods=["POST"])
def main():
	embedder = EmbedHandler()

	args = flask_req.args
	input_texts = args.get("texts")

	# Get the embeddings
	result = embedder.predict(input_texts)

	if not result.status_code == 200:
		print("Statuscode: ", result.status_code)
	else:
		result = result.json()["texts"]

	# Mini demo for similarity
	first_second_similarity = cosine_similarity([result[0]["embedding"]], [result[1]["embedding"]])
	first_third_similarity = cosine_similarity([result[0]["embedding"]], [result[2]["embedding"]])
	second_third_similarity = cosine_similarity([result[1]["embedding"]], [result[2]["embedding"]])

	print("0<->1: ",first_second_similarity,"| 0<->2: ", first_third_similarity,"| 1<->2: ", second_third_similarity)
	