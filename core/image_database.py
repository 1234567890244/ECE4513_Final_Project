import sqlite3
import pandas as pd
import os


def create_database():
    image_base_folder = "dataset/images/"
    csv_file_path = "dataset/emotions.csv"
    db = "dataset/emotions.sqlite"

    print("Reading CSV data...")
    try:
        df = pd.read_csv(csv_file_path, dtype={'set_id': str})
        print(f"Successfully read {len(df)} records")
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    print("Creating database...")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            set_id TEXT PRIMARY KEY,
            gender TEXT,
            age INTEGER,
            country TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            set_id TEXT REFERENCES faces(set_id),
            image_filename TEXT,
            image_path TEXT
        )
    ''')

    print("Inserting data...")
    df.to_sql('faces', conn, if_exists='replace', index=False)

    processed = 0
    errors = 0

    for sets in df['set_id']:
        class_folder = os.path.join(image_base_folder, sets)
        if not os.path.exists(class_folder):
            print(f"Missing folder for class: {sets}")
            continue
        try:
            file_list = os.listdir(class_folder)
            for filename in file_list:
                img_path = os.path.join(class_folder, filename)

                cursor.execute('''
                        INSERT INTO images (set_id, image_filename, image_path)
                        VALUES (?, ?, ?)
                    ''', (sets, filename, img_path))
                processed += 1

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            errors += 1

    conn.commit()

    print(f"Database created. Processed: {processed}, Errors: {errors}")

    cursor.execute("SELECT COUNT(*) FROM images")
    total_records = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM images WHERE image_path IS NOT NULL")
    records_with_images = cursor.fetchone()[0]

    conn.close()

    db_size = os.path.getsize(db) / 1024 / 1024

    print("=" * 50)
    print(f"Database file: {db}")
    print(f"Statistics:")
    print(f"   • Total records: {total_records}")
    print(f"   • Records with images: {records_with_images}")
    print(f"   • Database size: {db_size:.2f} MB")
    print("=" * 50)
