import sqlite3
from datetime import datetime


class UserDB:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT,
            register_date TEXT
        )''')
        self.conn.commit()

    def add_user(self, username, password):
        self.conn.execute('''
        INSERT INTO users (username, password, register_date)
        VALUES (?, ?, ?)''', (username, password, datetime.now()))
        self.conn.commit()

    def validate_user(self, username, password):
        cursor = self.conn.execute('''
        SELECT * FROM users WHERE username=? AND password=?''',
                                   (username, password))
        return cursor.fetchone() is not None


class MemeDB:
    pass
