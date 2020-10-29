# coding=utf-8
import ast
import json
import logging
import os
from abc import ABC

import torch
from transformers import AlbertForSequenceClassification
from transformers import BertTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning('Missing the setup_config.json file.')

        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path)
        elif self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"] == "sequence_classification":
                self.model = AlbertForSequenceClassification.from_pretrained(model_dir)
            # elif self.setup_config["mode"]== "question_answering":
            #     self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
            # elif self.setup_config["mode"]== "token_classification":
            #     self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            else:
                logger.warning('Missing the operation mode.')
        else:
            logger.warning('Missing the checkpoint or state_dict.')

        # if not os.path.isfile(os.path.join(model_dir, "vocab.*")):
        #     self.tokenizer = BertTokenizer.from_pretrained(self.setup_config["model_name"],
        #                                                    do_lower_case=self.setup_config["do_lower_case"])
        # else:
        self.tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=self.setup_config["do_lower_case"])

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # Question answering does not need the index_to_name.json file.
        if not self.setup_config["mode"] == "question_answering":
            if os.path.isfile(mapping_file_path):
                with open(mapping_file_path) as f:
                    self.mapping = json.load(f)
            else:
                logger.warning('Missing the index_to_name.json file.')

        self.initialized = True

    def preprocess(self, requests):
        """ Basic text preprocessing, based on the user's chocie of application mode.
        """
        input_batch = None
        for idx, data in enumerate(requests):
            text = data.get("data")
            if text is None:
                text = data.get("body")
            input_text = text.decode('utf-8')
            max_length = self.setup_config["max_length"]
            logger.info("Received text: '%s'", input_text)
            # preprocessing text for sequence_classification and token_classification.
            if self.setup_config["mode"] == "sequence_classification" or self.setup_config[
                "mode"] == "token_classification":
                inputs = self.tokenizer.encode_plus(input_text,
                                                    max_length=int(max_length),
                                                    pad_to_max_length=True,
                                                    add_special_tokens=True,
                                                    return_tensors='pt')

            input_ids = inputs["input_ids"].to(self.device)
            if input_ids.shape is not None:
                if input_batch is None:
                    input_batch = input_ids
                else:
                    input_batch = torch.cat((input_batch, input_ids), 0)
        return input_batch

    def inference(self, input_batch):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """

        inferences = []
        # Handling inference for sequence_classification.
        if self.setup_config["mode"] == "sequence_classification":
            predictions = self.model(input_batch)
            print("This the output size from the Seq classification model", predictions[0].size())
            print("This the output from the Seq classification model", predictions)

            num_rows, num_cols = predictions[0].shape
            for i in range(num_rows):
                out = predictions[0][i].unsqueeze(0)
                y_hat = out.argmax(1).item()
                predicted_idx = str(y_hat)
                inferences.append(self.mapping[predicted_idx])


        return inferences

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output



# torch-model-archiver --model-name BERTSeqClf_Torchscript --version 1.0 --serialized-file ./traced_bert.pt --handler ./clf_handler.py --extra-files "./classifier/finetuned/vocab.txt,./classifier/finetuned/config.json,./setup_config.json,./index_to_name.json"
# torchserve --start --model-store model_store --models my_tc=BERTSeqClf_Torchscript.mar

# curl -X POST "localhost:8081/models?model_name=BERTSeqClf_Torchscript&url=BERTSeqClf_Torchscript.mar&batch_size=1&max_batch_delay=5000&initial_workers=2&synchronous=true"
# curl -X POST http://127.0.0.1:8080/predictions/BERTSeqClf_Torchscript -T 凌云研发的国产两轮电动车怎么样，有什么惊喜？