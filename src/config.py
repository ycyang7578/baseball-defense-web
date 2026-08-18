"""共用設定：PostgreSQL 連線字串等專案唯一真相來源。

之前 DSN 字串在 11+ 個檔案（src/、scripts/、api/main.py）各自硬編碼一份，
改一個環境（換密碼、換 host）要記得改全部地方。統一從這裡 import。

部署雲端時設定 BASEBALL_DSN 環境變數（例如 Neon 給的連線字串）即可覆寫，
不用碰程式碼；本機沒設定時預設連本機 postgres。
"""
import os

DSN: str = os.environ.get(
    "BASEBALL_DSN",
    "host=localhost dbname=baseball user=postgres password=postgres",
)
