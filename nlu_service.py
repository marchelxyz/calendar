"""Сервис обработки естественного языка для извлечения данных о событиях"""
from openai import OpenAI
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from config import Config
import json
import logging
import pytz
from dateutil import parser

logger = logging.getLogger(__name__)

class NLUService:
    """Сервис для обработки текста и извлечения информации о событиях"""
    
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.timezone = pytz.timezone(Config.TIMEZONE)
    
    def _get_current_datetime(self) -> datetime:
        """Получение текущей даты и времени в нужном часовом поясе"""
        return datetime.now(self.timezone)
    
    def _create_prompt(self, text: str) -> str:
        """Создание промпта для LLM"""
        current_datetime = self._get_current_datetime()
        current_date_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        prompt = f"""Ты помощник для создания событий в календаре. Пользователь отправил голосовое сообщение, которое было транскрибировано в текст.

Текущая дата и время: {current_date_str} (часовой пояс: {Config.TIMEZONE})

Текст пользователя: "{text}"

Твоя задача - извлечь из текста информацию о событии и вернуть JSON со следующей структурой:
{{
    "action": "create_event" | "delete_event" | "update_event",
    "summary": "Название события",
    "start_datetime": "YYYY-MM-DDTHH:MM:SS",
    "duration_minutes": 60,
    "description": "Описание (опционально)"
}}

Правила:
1. Если пользователь говорит "завтра", "послезавтра", "через 3 дня" - вычисли правильную дату относительно текущей даты
2. Если указано время без даты (например, "в 3 часа дня"), используй сегодняшнюю дату, если событие еще не прошло, иначе завтрашнюю
3. Если время не указано, используй 12:00 по умолчанию
4. Если длительность не указана, используй 60 минут по умолчанию
5. Если пользователь просит удалить или изменить событие, укажи action соответственно
6. Всегда возвращай валидный JSON, без дополнительного текста

Примеры:
- "Поставь встречу с клиентом на завтра в 15:00" -> {{"action": "create_event", "summary": "Встреча с клиентом", "start_datetime": "2025-01-15T15:00:00", "duration_minutes": 60}}
- "Созвон с командой послезавтра в 10 утра на час" -> {{"action": "create_event", "summary": "Созвон с командой", "start_datetime": "2025-01-16T10:00:00", "duration_minutes": 60}}
- "Напомни мне про презентацию через 2 дня в 14:30" -> {{"action": "create_event", "summary": "Презентация", "start_datetime": "2025-01-16T14:30:00", "duration_minutes": 60}}

Верни только JSON, без дополнительных комментариев:"""
        
        return prompt
    
    async def extract_event_info(self, text: str) -> Dict[str, Any]:
        """
        Извлечение информации о событии из текста
        
        Args:
            text: Транскрибированный текст
            
        Returns:
            Словарь с информацией о событии
        """
        try:
            prompt = self._create_prompt(text)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Ты помощник для создания событий в календаре. Всегда возвращай только валидный JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Парсим дату и время
            if "start_datetime" in result:
                dt_str = result["start_datetime"]
                # Если дата без часового пояса, добавляем его
                try:
                    dt = parser.parse(dt_str)
                    if dt.tzinfo is None:
                        dt = self.timezone.localize(dt)
                    result["start_datetime"] = dt
                except Exception as e:
                    logger.error(f"Ошибка парсинга даты {dt_str}: {e}")
                    # Используем текущее время + 1 день как fallback
                    result["start_datetime"] = self._get_current_datetime() + timedelta(days=1)
            
            # Устанавливаем значения по умолчанию
            result.setdefault("action", "create_event")
            result.setdefault("duration_minutes", 60)
            result.setdefault("description", None)
            
            logger.info(f"Извлечена информация о событии: {result}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON от LLM: {e}")
            raise ValueError("Не удалось обработать запрос. Попробуйте сформулировать иначе.")
        except Exception as e:
            logger.error(f"Ошибка обработки текста: {e}")
            raise
