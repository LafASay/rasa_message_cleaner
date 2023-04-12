from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
import re
# TODO: Correctly register your component with its type
@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER], is_trainable=True
)

class CustomNLUComponent(GraphComponent):
    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        # TODO: Implement this
        ...

    def train(self, training_data: TrainingData) -> Resource:
        # TODO: Implement this if your component requires training
        ...

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        # TODO: Implement this if your component augments the training data with
        #       tokens or message features which are used by other components
        #       during training.
        ...

        return training_data

    # def process(self, messages: List[Message]) -> List[Message]:
    #     # TODO: This is the method which Rasa Open Source will call during inference.
    #     RE_WORD_COMMA = re.compile (r'([^0-9 ,]),')
    #     RE_COMMA_WORD = re.compile (r',([^0-9 ,])')
    #     RE_THREE_LETTERS = re.compile (r'(\D)\1{3,}')
    #     RE_TWO_SPACES = re.compile (r'( )\1{1,}')        
    #     for message in messages:
    #         try:
    #             message.data["text"] = message.data["text"].lower().replace('ё', 'е')
    #             message.data["text"] = re.sub(RE_WORD_COMMA, r'\1 ,', message.data["text"])
    #             message.data["text"] = re.sub(RE_COMMA_WORD, r', \1', message.data["text"])
    #             message.data["text"] = re.sub(RE_THREE_LETTERS, r'\1\1', message.data["text"])
    #             message.data["text"] = re.sub(RE_TWO_SPACES, r' ', message)
    #             print(message.data["text"])
    #         except:
    #             print("error")
    #     return messages

    def process(self, messages: List[Message]) -> List[Message]:
        # TODO: This is the method which Rasa Open Source will call during inference.
        # RE_WORD_COMMA = re.compile (r'([^0-9 ,]),')
        # RE_COMMA_WORD = re.compile (r',([^0-9 ,])')
        # RE_THREE_LETTERS = re.compile (r'(\D)\1{3,}')
        # RE_TWO_SPACES = re.compile (r'( )\1{1,}')        
        for message in messages:
            try:
                # в нижний регистр, замена ё на е
                message.data["text"] = message.data["text"].lower().replace('ё', 'е')
                # поиск и замена на пробел всех не нужных символов
                message.data["text"] = re.sub(r"[^\w.,?!+-]+", " ", message.data["text"])
                # поиск ненужного пробела между спецсимволами и его удаление
                message.data["text"] = re.sub(r'([.,?!+-])\s+([.,?!+-])', r'\1\2', message.data["text"])
                # поиск спецсимвола за которым следует буква и вставка пробела между ними
                message.data["text"] = re.sub(r"([.,?!+-])([а-яa-z])", r"\1 \2", message.data["text"])
                # поиск более одного одинакового спецсимвола подряд и сокращение до одного
                message.data["text"] = re.sub(r"([.,?!+-])\1+", r"\1", message.data["text"])
                # поиск более двух одинаковых букв и сокращение до двух
                message.data["text"] = re.sub(r"(\w)\1{1,}", r"\1\1", message.data["text"])
                # поиск слова за которым идет пробел и спецсимвол и удаление ненужного пробела
                message.data["text"] = re.sub(r"(\w)\s+([!?,.])", r"\1\2", message.data["text"])
                # поиск запятой, точки или пробела в конце и его удаление
                message.data["text"] = re.sub(r"[.,\s]+$", "", message.data["text"])
                # поиск более двух пробелов подряд и сокращение до одного
                message.data["text"] = re.sub(r"\s{2,}", " ", message.data["text"])
                print(f'clean_message: {message.data["text"]}')
            except:
                print("clean_message: error")
        return messages
