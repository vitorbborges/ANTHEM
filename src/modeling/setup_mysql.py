# setup_mysql_study.py - Create/recreate Optuna study in MySQL
import os
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import optuna

from src.modeling.utils import log_message


def setup_mysql_study():
    """Set up MySQL study for distributed optimization."""

    # Configuration
    DB_USER = "optuna_user"
    DB_PASS = "anthem1234"
    DB_NAME = "optuna_db"
    STUDY_NAME = "co2_prediction"
    STORAGE_URL = f"mysql://{DB_USER}:{DB_PASS}@localhost/{DB_NAME}"

    log_message("🔧 Setting up MySQL study...")
    log_message(f"Database: {DB_NAME}")
    log_message(f"Study: {STUDY_NAME}")

    try:
        # Install required packages
        log_message("📦 Installing/upgrading required packages...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "optuna", "PyMySQL"],
            check=True,
            capture_output=True,
        )

        # Delete existing study if it exists
        log_message(f"🔄 Deleting existing study '{STUDY_NAME}' (if any)...")
        try:
            optuna.delete_study(study_name=STUDY_NAME, storage=STORAGE_URL)
            log_message("✅ Existing study deleted")
        except Exception as e:
            log_message(f"ℹ️  No existing study found (this is normal): {str(e)}")

        # Create new study
        log_message(f"✨ Creating fresh study '{STUDY_NAME}'...")
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=STORAGE_URL,
            direction="minimize",
            load_if_exists=False,
        )

        log_message("✅ Study created successfully!")
        log_message(f"📊 Study info: {len(study.trials)} trials")

        # Test the connection
        log_message("🧪 Testing study access...")
        test_study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
        log_message(
            f"✅ Connection test passed! Study has {len(test_study.trials)} trials"
        )

        log_message("🚀 Ready to run optimization!")
        log_message("📝 Example commands:")
        log_message(f"   # Single process:")
        log_message(
            f'   python -m src.modeling.main --storage mysql --mysql-url "{STORAGE_URL}"'
        )
        log_message(f"   # Parallel:")
        log_message(
            f'   python -m src.modeling.main --parallel --workers 4 --storage mysql --mysql-url "{STORAGE_URL}"'
        )
        log_message(f"   # Dashboard:")
        log_message(f'   optuna-dashboard "{STORAGE_URL}"')

        return STORAGE_URL

    except subprocess.CalledProcessError as e:
        log_message(f"❌ Failed to install packages: {e}")
        return None
    except Exception as e:
        log_message(f"❌ Failed to set up study: {str(e)}")
        log_message("💡 Make sure MySQL is running and credentials are correct")
        log_message("💡 Try: sudo systemctl start mysql")
        return None


def test_mysql_connection():
    """Test MySQL connection before creating study."""
    import pymysql

    try:
        connection = pymysql.connect(
            host="localhost",
            user="optuna_user",
            password="anthem1234",
            database="optuna_db",
        )
        connection.close()
        log_message("✅ MySQL connection test passed")
        return True
    except Exception as e:
        log_message(f"❌ MySQL connection failed: {str(e)}")
        log_message("💡 Make sure MySQL is running: sudo systemctl start mysql")
        log_message("💡 Make sure database and user exist")
        return False


if __name__ == "__main__":
    log_message("🚀 MySQL Study Setup for CO2 Prediction")
    log_message("=" * 50)

    # Test connection first
    if test_mysql_connection():
        storage_url = setup_mysql_study()
        if storage_url:
            log_message(
                "\n🎉 Setup complete! You can now run distributed optimization."
            )
        else:
            log_message("\n❌ Setup failed. Please check the errors above.")
    else:
        log_message("\n❌ Cannot connect to MySQL. Please fix connection issues first.")
