from trulens_eval import Tru
from dotenv import load_dotenv
import os

load_dotenv()

tru = Tru(database_url=os.getenv('TRULENS_DB_URL'))
# tru.reset_database()
tru.run_dashboard()