from cryptography.fernet import Fernet
import uuid

# User data and consent stores
user_data_store = {}
user_consent_store = {}

# Generate encryption key and cipher suite
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    return cipher_suite.decrypt(encrypted_data).decode()

def generate_anonymous_id():
    return str(uuid.uuid4())

def store_user_data(anonymous_id, data):
    encrypted_data = encrypt_data(data)
    user_data_store[anonymous_id] = encrypted_data

def retrieve_user_data(anonymous_id):
    encrypted_data = user_data_store.get(anonymous_id)
    if encrypted_data:
        return decrypt_data(encrypted_data)
    return None

def store_user_consent(anonymous_id, consent):
    user_consent_store[anonymous_id] = consent