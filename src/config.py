"""共用設定：PostgreSQL 連線字串等專案唯一真相來源。

之前 DSN 字串在 11+ 個檔案（src/、scripts/、api/main.py）各自硬編碼一份，
改一個環境（換密碼、換 host）要記得改全部地方。統一從這裡 import。
"""
DSN = "host=localhost dbname=baseball user=postgres password=postgres"
