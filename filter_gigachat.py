"""
Обёртка LLM-фильтра поверх GigaChat (Сбер).
Вся логика промпта/валидации — в llm_filter_core.py, здесь только
получение OAuth-токена и вызов chat/completions, адаптированные
под интерфейс call_fn(system_prompt, user_prompt) -> str.

Важные особенности GigaChat, отличающие эту обёртку от filter_gpt.py:

1. Нет отдельного "режима гарантированного JSON" (response_format) —
   полагаемся полностью на промпт и на strip_code_fences/json.loads
   из llm_filter_core, как и без response_format. Это ровно то же самое,
   что и раньше, так что дополнительных изменений тут не нужно.

2. Аутентификация двухэтапная: сначала по авторизационному ключу
   (Base64-строка client_id:client_secret из личного кабинета Сбера)
   получаем Bearer-токен через OAuth, затем этим токеном подписываем
   запросы к chat/completions. Токен живёт ограниченное время —
   кэшируем и обновляем по истечении, а не запрашиваем каждый раз заново.

3. Сертификат: у GigaChat самоподписанный/не входящий в стандартные
   доверенные цепочки сертификат (нужен сертификат НУЦ Минцифры для
   полной проверки). Для простоты по умолчанию verify=False — для
   продакшена лучше один раз скачать и указать сертификат Минцифры
   через параметр ca_bundle, а не отключать проверку насовсем.
"""

import time
import uuid
import requests
import urllib3

from llm_filter_core import filter_candidates as _filter_candidates

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


class GigaChatClient:
    def __init__(
        self,
        authorization_key: str,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "GigaChat-2",
        ca_bundle: str | None = None,
    ):
        """
        authorization_key: строка из личного кабинета Сбера ("Authorization key",
            уже в формате Base64, просто подставляется как есть).
        scope: GIGACHAT_API_PERS для физлиц, GIGACHAT_API_B2B / GIGACHAT_API_CORP
            для юрлиц — зависит от того, какой у вас доступ.
        model: актуальные модели второго поколения — GigaChat-2,
            GigaChat-2-Pro, GigaChat-2-Max (первое поколение автоматически
            перенаправляется на второе, но лучше указывать явно).
        ca_bundle: путь к сертификату НУЦ Минцифры, если есть. Если None —
            проверка SSL отключается (verify=False), что работает, но
            менее безопасно для продакшена.
        """
        self.authorization_key = authorization_key
        self.scope = scope
        self.model = model
        self.verify = ca_bundle if ca_bundle else False
        self._token = None
        self._token_expires_at = 0.0

    def _fetch_token(self) -> str:
        response = requests.post(
            OAUTH_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {self.authorization_key}",
            },
            data={"scope": self.scope},
            verify=self.verify,
            timeout=30,
        )
        if response.status_code == 401:
            raise RuntimeError(
                "GigaChat OAuth: 401 Unauthorized. Проверьте authorization_key "
                "и scope (GIGACHAT_API_PERS/B2B/CORP должен соответствовать "
                "вашему тарифу в личном кабинете Сбера)."
            )
        response.raise_for_status()
        data = response.json()

        self._token = data["access_token"]
        expires_at = data["expires_at"]
        # у Сбера expires_at обычно в миллисекундах с эпохи
        self._token_expires_at = expires_at / 1000 if expires_at > 1e12 else expires_at
        return self._token

    def _get_valid_token(self) -> str:
        # обновляем заранее (за 30 сек до истечения), а не строго в момент истечения
        if self._token is None or time.time() >= self._token_expires_at - 30:
            return self._fetch_token()
        return self._token

    def call(self, system_prompt: str, user_prompt: str) -> str:
        token = self._get_valid_token()

        def _do_request(bearer_token: str):
            return requests.post(
                CHAT_URL,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {bearer_token}",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                },
                verify=self.verify,
                timeout=60,
            )

        response = _do_request(token)
        if response.status_code == 401:
            # токен мог истечь между проверкой и запросом — обновляем один раз и повторяем
            token = self._fetch_token()
            response = _do_request(token)

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def list_available_models(self) -> list[str]:
        """
        Возвращает список ID моделей, доступных именно этому ключу/scope —
        без единого потраченного токена (не chat/completions, а служебный
        эндпоинт). Самый надёжный способ проверить, доступен ли конкретной
        учётке GigaChat-2-Pro/Max, вместо того чтобы гадать по документации
        или тратить токены на пробный вызов, который может просто упасть
        с ошибкой прав доступа.
        """
        token = self._get_valid_token()
        response = requests.get(
            "https://gigachat.devices.sberbank.ru/api/v1/models",
            headers={"Authorization": f"Bearer {token}"},
            verify=self.verify,
            timeout=30,
        )
        response.raise_for_status()
        return [m["id"] for m in response.json()["data"]]


def filter_candidates(client: GigaChatClient, lemma: str, candidates: list[dict], max_retries: int = 2) -> list[dict]:
    return _filter_candidates(client.call, lemma, candidates, max_retries=max_retries)


if __name__ == "__main__":
    import os
    from generator import gen_candidates

    client = GigaChatClient(
        # authorization_key=os.environ["GIGACHAT_AUTH_KEY"],
        authorization_key="MDE5ZjE0NzgtZDQxNy03MzZkLTgzZDQtYTQyZjViMWE1MWExOjFhOGYxZTI0LTE2MzMtNDMzNS1hNzdhLTMyZjdlMmMwMGRlZA==",
        scope="GIGACHAT_API_PERS",
        model="GigaChat-2",
    )

    for word in ["земля", "рука", "дождь"]:
        candidates = gen_candidates(word)
        filtered = filter_candidates(client, word, candidates)
        print(f"\n{word}:")
        for r in filtered:
            status = "✓" if r["exists"] else ("✗" if r["exists"] is False else "?")
            print(f"  {status} {r['form']:<16} connotation={r['connotation']}")