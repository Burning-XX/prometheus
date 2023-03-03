"""
   ___                            _        _   _                   _                                        _          
  / __\___  _ __ ___  _ __  _   _| |_ __ _| |_(_) ___  _ __   __ _| |   ___  ___ ___  _ __   ___  _ __ ___ (_) ___ ___ 
 / /  / _ \| '_ ` _ \| '_ \| | | | __/ _` | __| |/ _ \| '_ \ / _` | |  / _ \/ __/ _ \| '_ \ / _ \| '_ ` _ \| |/ __/ __|
/ /__| (_) | | | | | | |_) | |_| | || (_| | |_| | (_) | | | | (_| | | |  __/ (_| (_) | | | | (_) | | | | | | | (__\__ \
\____/\___/|_| |_| |_| .__/ \__,_|\__\__,_|\__|_|\___/|_| |_|\__,_|_|  \___|\___\___/|_| |_|\___/|_| |_| |_|_|\___|___/
                     |_|                                                                                               

计算经济学数据处理工具箱 API
WSGI.PY
WSGI配置

官方文档:
https://docs.djangoproject.com/en/3.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'zyp.settings')

application = get_wsgi_application()
