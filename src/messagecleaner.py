import typing
from typing import Any, Dict, List, Text, Type, Union

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import TEXT
from rasa.nlu.utils.spacy_utils import SpacyModel, SpacyNLP
from rasa.shared.nlu.training_data.message import Message
from spellchecker import SpellChecker
import re
from rasa.nlu.constants import DENSE_FEATURIZABLE_ATTRIBUTES, SPACY_DOCS
import json


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
    is_trainable=False,
    model_from="SpacyNLP",
)
class CustomNLUComponent(GraphComponent):
    """Entity extractor which uses SpaCy."""

    def cleaner(string):
        # в нижний регистр, замена ё на е
        string = string.lower().replace('ё', 'е').replace('<q>', ' ').replace('</q>', ' ').replace('<b>', ' ').replace('</b>', ' ')
        # поиск и замена запятой между цифрами на точку
        string = re.sub(r'(\d),(\d)', r'\1.\2', string)
        # поиск и замена на пробел всех не нужных символов
        string = re.sub(r"[^\w%.,/=#?&:@*!+-]+", " ", string)
        # поиск ненужного пробела между спецсимволами и его удаление
        string = re.sub(r'([%.,=?:!+-])\s+([%.,=?:!+-])', r'\1\2', string)
        # поиск спецсимвола за которым следует буква и вставка пробела между ними
        string = re.sub(r"([%.,=?:!+])([а-я])", r"\1 \2", string)
        # поиск более одного одинакового спецсимвола подряд и сокращение до одного
        string = re.sub(r"([%.,=?:!+-])\1+", r"\1", string)
        # поиск более двух одинаковых букв и сокращение до двух
        string = re.sub(r'([а-я])\1{2,}', r'\1\1', string)
        # поиск слова за которым идет пробел и спецсимвол и удаление ненужного пробела
        string = re.sub(r"(\w)\s+([!?,.])", r"\1\2", string)
        # поиск спецсимволов или пробела в конце и его удаление
        string = re.sub(r"[-.,!:\s]+$", "", string)
        # поиск более двух пробелов подряд и сокращение до одного
        string = re.sub(r"\s{2,}", " ", string)
        # удалить пробел или точку в начале строки
        string = re.sub(r'^[-.,!%?\s]+', '', string)
        return string

    def is_cyrillic_only(string):
        cyrillic_pattern = re.compile(r'^[а-яА-Я]+$')
        return bool(cyrillic_pattern.match(string))

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [SpacyNLP]

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

    @staticmethod
    def required_packages() -> List[Text]:
        """Lists required dependencies (see parent class for full docstring)."""
        return ["spacy"]

    def is_special_string(string):
        # Используем регулярное выражение для проверки строки
        pattern = r"^[.:_, ;»«&…!'\"]+$"  # Это регулярное выражение соответствует только запятым, точкам, восклицательным знакам и кавычкам
        return bool(re.match(pattern, string))

    def spell_correct(string, spell):
        # Условие для проверки слова на правильность орфографии
        if string.isalpha() and len(string) > 3 and CustomNLUComponent.is_cyrillic_only(string):
            print(f"CustomNLUComponent.fix_spacing_typo(string) = {CustomNLUComponent.fix_spacing_typo(string)}")
            if string != CustomNLUComponent.fix_spacing_typo(string):
                return CustomNLUComponent.fix_spacing_typo(string)
            else:
                print(f"spell.correction(string) = {spell.correction(string)}")
                return spell.correction(string)

    def spell_correct(tokens):
        # Создаем экземпляр SpellChecker
        spell = SpellChecker(language='wmru')

        corrected_words = []
        # Проверяем каждое слово на правильность орфографии
        for token in tokens:
            cyrillic_pattern = re.compile(r'^[а-яА-Я]+$')
            if cyrillic_pattern.match(token) and token.isalpha() and len(token) > 3:
                # Проверяем, является ли слово правильным
                if spell.correction(token) != token:
                    # Если слово неправильное, исправляем его
                    if spell.correction(token) != None:
                        corrected_words.append(spell.correction(token))
                    else:
                        corrected_words.append(token)
                else:
                    # Если слово правильное, оставляем его без изменений
                    corrected_words.append(token)
            else:
                # Если слово содержит не только буквы, оставляем его без изменений
                corrected_words.append(token)

        # Собираем исправленные слова обратно в сообщение
        corrected_string = ' '.join(corrected_words)

        return corrected_string

    def process(self, messages: List[Message], model: SpacyModel) -> List[Message]:
        for message in messages:
            if message.get(TEXT) != '':
                spacy_nlp = model.model
                string = message.get(TEXT)
                string = CustomNLUComponent.cleaner(string)

                doc = spacy_nlp(string)
                tokens = []
                for token in doc:
                    tokens.append(token.text)
                print(f'tokens = {tokens}')

                string = CustomNLUComponent.spell_correct(tokens)
                string = CustomNLUComponent.cleaner(string)

                if message.get(TEXT) != string and len(string) > 0:
                    message.set(
                        SPACY_DOCS[TEXT],
                        spacy_nlp(string)
                    )

                print(f"{{'TEXT': '{message.get(TEXT)}', 'SPACY_TEXT': '{message.get(SPACY_DOCS[TEXT])}'}}")

        return messages
