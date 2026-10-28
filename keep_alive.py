
import os
import time
import logging
import threading

import requests
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SELF_PING_INTERVAL = 10 * 60       # 10 minutes
DATABASE_CHECK_INTERVAL = 10 * 60 * 60  # 10 hours

# local test timers
# SELF_PING_INTERVAL = 10       # 10 seconds
# DATABASE_CHECK_INTERVAL = 30  # 30 seconds

stop_event = threading.Event()


def ping_render():
    url = os.getenv("BACKEND_HEALTH_URL")

    if not url:
        logger.warning("BACKEND_HEALTH_URL is not configured")
        return

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        logger.info("Render self-ping successful: %s", response.status_code)
    except requests.RequestException as exc:
        logger.warning("Render self-ping failed: %s", exc)


def check_qdrant():
    client = None

    try:
        client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
            timeout=20,
        )

        client.get_collections()
        logger.info("Qdrant API check successful")

    except Exception:
        logger.exception("Qdrant API check failed")

    finally:
        if client is not None:
            client.close()


def check_auradb():
    driver = None

    try:
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(
                os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"],
            ),
        )

        with driver.session() as session:
            record = session.run(
                "RETURN datetime() AS checked_at"
            ).single()

            logger.info(
                "AuraDB query successful: %s",
                record["checked_at"],
            )

    except Exception:
        logger.exception("AuraDB query failed")

    finally:
        if driver is not None:
            driver.close()


def maintenance_loop():
    last_db_check = 0.0

    while not stop_event.is_set():
        # Wait before the first self-ping.
        if stop_event.wait(SELF_PING_INTERVAL):
            break

        ping_render()

        # Run database checks once every 24 hours.
        if time.monotonic() - last_db_check >= DATABASE_CHECK_INTERVAL:
            check_qdrant()
            check_auradb()
            last_db_check = time.monotonic()


def start_keep_alive():
    thread = threading.Thread(
        target=maintenance_loop,
        name="keep-alive",
        daemon=True,
    )
    thread.start()
    logger.info("Keep-alive loop started")
