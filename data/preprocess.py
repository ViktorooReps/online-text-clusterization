import json


def is_corrupted(text: str) -> bool:
    if not len(text):
        return True

    if 'javascript' in text or 'JavaScript' in text and len(text) < 200:
        return True

    if text == 'Печать Нашли ошибку в тексте? Выделите ее и нажмите Ctrl + Enter':
        return True

    if text == 'Чтобы сообщить нам об опечатке, выделите ее мышкой и нажмите Ctrl+Enter':
        return True

    if text == 'Лента новостей Лента новостей':
        return True

    if text == 'В мире Добавил: Oksana Сегодня, 02:00':
        return True

    if text == 'Tab 2 content goes here... Tab 3 content goes here...':
        return True

    if text == 'Криштиану Роналду / Фото: © Michael Regan / Staff / Getty Images Sport / Gettyimages.ru':
        return True

    if text == 'Мы говорим вам правду. Вы решаете, что с ней делать.':
        return True

    if text == '× Вы можете редактировать свой комментарий только в течении 5 минут':
        return True

    if text == 'Вы собираетесь перейти по внешней ссылке: Вы действительно хотите перейти по ссылке?':
        return True

    if text == 'Oh boy!':
        return True

    if text == 'Авто Добавил: tantan61 Вчера, 13:30':
        return True

    if text == '24.04.18 7:38 текст: Ирина Клячина фото: скриншот Яндекс.Картинки 49':
        return True

    return False


if __name__ == '__main__':
    with open('data/dev-dataset-task2022-04.json') as f:
        data = json.load(f)

    preprocessed_data = []
    for example_text, example_category in data:
        if is_corrupted(example_text):
            print('DELETED: ' + example_text)
            continue

        preprocessed_data.append((example_text, example_category))

    # deduplicate
    preprocessed_data = set(preprocessed_data)

    with open('data/dev-dataset-task2022-04_preprocessed.json', 'w') as f:
        json.dump(list(preprocessed_data), f, ensure_ascii=False, indent=4)
