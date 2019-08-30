import os
import socket


def send_noti_to_telegram(message):
    try:
        import telegram
    except ImportError:
        print('telegram package is not installed. ' +
              'please install the package by executing $ pip install python-telegram-bot')
        return
    except Exception as e:
        print('sending message failed', e)
        return

    try:
        token = os.environ['TELEGRAM_TOKEN']
        chat_id = os.environ['TELEGRAM_CHAT_ID']
        bot = telegram.Bot(token=token)
        bot.sendMessage(chat_id=chat_id, text=f'[{socket.gethostname()}]\n{message}')
    except KeyError:
        print('TELEGRAM_TOKEN or TELEGRAM_CHAT_ID is not set')
        pass
    except Exception as e:
        print('sending message failed', e)
        return
