import sqlite3


DB_PATH = 'face_data.db'

def fetch_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Fetch data from landmarks table
    c.execute("SELECT * FROM landmarks")
    landmarks_data = c.fetchall()
    print("Landmarks Data:")
    for row in landmarks_data:
        print(row)

    # Fetch data from distances table
    c.execute("SELECT * FROM distances")
    distances_data = c.fetchall()
    print("\nDistances Data:")
    for row in distances_data:
        print(row)

    conn.close()

if __name__ == '__main__':
    fetch_data()
