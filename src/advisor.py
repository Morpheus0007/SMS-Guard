"""Actionable user advice keyed by canonical FEATURE_NAMES and domain metadata."""

from __future__ import annotations

_ADVICE_BY_FEATURE: dict[str, str] = {
    "asks_code_or_password": (
        "Никому не сообщайте одноразовые SMS-коды, PIN-коды, пароли и CVV — "
        "настоящие сотрудники банков и операторов никогда их не запрашивают."
    ),
    "has_brand_spoof_domain": (
        "Не переходите по подозрительной ссылке: домен имитирует известный бренд "
        "или сокращатель и не входит в официальный белый список."
    ),
    "has_link": (
        "Не вводите личные и банковские данные по ссылкам из SMS; откройте официальное "
        "приложение банка или наберите официальный адрес сайта вручную."
    ),
    "threatens_block": (
        "Не поддавайтесь на угрозы блокировки карты или SIM-карты: проверьте статус "
        "счёта самостоятельно через официальное приложение или по номеру на карте."
    ),
    "asks_money_transfer": (
        "Не переводите деньги незнакомым людям, даже если просят «вернуть ошибочный перевод», "
        "«оплатить доставку приза» или срочно «скинуть в долг» — позвоните близкому на старый номер."
    ),
    "promises_prize": (
        "Относитесь скептически к неожиданным выигрышам, призам и бесплатным гигабайтам: "
        "мошенники используют акции для кражи аккаунтов и денег."
    ),
    "asks_callback": (
        "Не перезванивайте на неизвестные и зарубежные номера из SMS. Используйте только "
        "официальный номер поддержки с сайта банка или с обратной стороны карты."
    ),
    "has_phone_number": (
        "Сверьте указанный номер с официальными контактами организации перед тем, как звонить или писать."
    ),
    "has_urgency": (
        "Игнорируйте искусственную срочность («через 24 часа», «тез арада»): это приём "
        "давления, чтобы вы не успели перепроверить информацию."
    ),
}


def get_advice(
    risk: str,
    triggered_feature_keys: list[str],
    domains: list[dict[str, object]],
) -> list[str]:
    """Return practical Russian safety recommendations based on canonical feature keys."""
    advice: list[str] = []
    seen: set[str] = set()

    def _add(item: str) -> None:
        if item not in seen:
            seen.add(item)
            advice.append(item)

    for key in triggered_feature_keys:
        if key in _ADVICE_BY_FEATURE:
            _add(_ADVICE_BY_FEATURE[key])

    if any(bool(d.get("is_brand_spoof")) for d in domains):
        _add(_ADVICE_BY_FEATURE["has_brand_spoof_domain"])

    if not advice:
        if risk == "high":
            _add(
                "Высокий риск мошенничества: не выполняйте инструкции из сообщения, "
                "не переходите по ссылкам и удалите SMS."
            )
        elif risk == "medium":
            _add(
                "Сообщение содержит настораживающие признаки: перепроверьте отправителя "
                "через официальное приложение банка или оператора."
            )
        else:
            if any(bool(d.get("is_official")) for d in domains):
                _add(
                    "Домен входит в официальный белый список, однако всегда проверяйте адресную "
                    "строку браузера перед вводом данных."
                )
            else:
                _add(
                    "Явных признаков фишинга не обнаружено. Все равно сохраняйте осторожность "
                    "и никогда не передавайте коды из SMS третьим лицам."
                )

    return advice
