import sqlite3
from PIL import Image
import io
import os


class DatabaseManager:
    def __init__(self, db_path: str = 'default.sqlite'):

        self.db_path = db_path
        self.conn = None
        self.cursor = None

        self._connect()

    def _connect(self):
        try:
            if not os.path.exists(os.path.dirname(self.db_path)):
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                self._initialize_db()

            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()

        except Exception as e:
            raise ConnectionError(f"failed to connect database {self.db_path}: {str(e)}")

    def _initialize_db(self):
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            tags TEXT,
            image_data BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.conn.commit()

    def switch_database(self, new_db_path: str):
        self.close()
        self.db_path = new_db_path
        self._connect()

    def get_all_images_metadata(self):
        self.cursor.execute("SELECT image_path FROM images")
        return self.cursor.fetchall()

    def get_image_by_id(self, image_id):
        self.cursor.execute("SELECT image_path FROM images WHERE id=?", (image_id,))
        image_path = self.cursor.fetchone()[0]
        return image_path

    def search_images(self, keyword):
        self.cursor.execute(
            "SELECT id, image_path FROM images WHERE image_path LIKE ?",
            (f'%{keyword}%',)
        )
        return self.cursor.fetchall()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.conn.close()
