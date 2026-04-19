"""E2E RAG evaluation gate — release-readiness check.

Seeds a fixture collection with 50 known chunks, runs the eval dataset
through RAGEvaluator, and asserts acceptance thresholds:
  Recall@5 ≥ 0.7
  MRR     ≥ 0.5

If metrics fall below thresholds the test goes red — this is a release gate.

Requires: running Qdrant on localhost:6333 and BGE-M3 model cached.
Run:  python -m pytest tests/e2e/test_rag_eval.py -v -s
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ── Fixture chunks: 50 chunks across 5 models / 4 vendors ───────────────────

_COLLECTION = f"_test_rag_eval_{uuid.uuid4().hex[:8]}"

_CHUNKS = [
    # --- Kyocera TASKalfa 3253ci (10 chunks) ---
    {
        "text": "Error Code C6000 — Abnormal fuser heater temperature. The fuser unit fails to reach target temperature within 30 seconds. Replace the fuser thermistor or the fuser heater lamp.",
        "vendor": "Kyocera", "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["C6000"], "doc_type": "service_manual",
    },
    {
        "text": "Error Code C6020 — Fuser overheat. The fuser temperature exceeds the upper safety limit. Check thermistor wiring and fuser unit connector CN201.",
        "vendor": "Kyocera", "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["C6020"], "doc_type": "service_manual",
    },
    {
        "text": "Error Code C0800 — Image density sensor malfunction. Clean the ID sensor lens. If the error persists, replace the ID sensor assembly.",
        "vendor": "Kyocera", "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["C0800"], "doc_type": "service_manual",
    },
    {
        "text": "The drum unit (OPC drum) should be replaced every 200,000 pages. Check drum surface for scratches or coating damage.",
        "vendor": "Kyocera", "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Developer unit replacement procedure: 1) Open front cover. 2) Remove developer unit by pulling the green handle. 3) Install the new developer unit.",
        "vendor": "Kyocera", "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Paper feed roller replacement: Remove cassette tray. Release the roller retainer clips. Replace the pickup roller and feed roller set.",
        "vendor": "Kyocera", "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Periodic maintenance table for TASKalfa 3253ci. Fuser replacement: 600K pages. Drum: 200K. Developer: 600K. Transfer belt: 300K.",
        "vendor": "Kyocera", "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Scanner CIS unit calibration for TASKalfa 3253ci. Run auto calibration from service mode 8001. Replace CIS if streaks persist.",
        "vendor": "Kyocera", "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Error Code J0510 — Paper jam at the transfer unit area. Remove jammed paper from the transfer belt section. Check transfer belt tension.",
        "vendor": "Kyocera", "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["J0510"], "doc_type": "service_manual",
    },
    {
        "text": "Error Code E00010 — Main motor failure. Check motor connector CN101 and PCB relay. Replace main motor assembly if drive signal is absent.",
        "vendor": "Kyocera", "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["E00010"], "doc_type": "service_manual",
    },
    # --- Kyocera ECOSYS M2040dn (7 chunks) ---
    {
        "text": "Error Code C6000 on ECOSYS M2040dn — Fuser heater error. This model uses a thin-film fuser. Replace the fuser unit assembly (FK-1150).",
        "vendor": "Kyocera", "model": "Kyocera ECOSYS M2040dn",
        "error_codes": ["C6000"], "doc_type": "service_manual",
    },
    {
        "text": "ECOSYS M2040dn — drum replacement procedure. Use DK-1150 drum kit. Expected yield: 100,000 pages.",
        "vendor": "Kyocera", "model": "Kyocera ECOSYS M2040dn",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Toner supply error on M2040dn. Error code C7990 — toner sensor detects empty cartridge. Replace TK-1170 toner cartridge.",
        "vendor": "Kyocera", "model": "Kyocera ECOSYS M2040dn",
        "error_codes": ["C7990"], "doc_type": "service_manual",
    },
    {
        "text": "M2040dn network configuration: access embedded web server via https://<ip>. Default admin password: Admin00. Configure SMTP for scan-to-email.",
        "vendor": "Kyocera", "model": "Kyocera ECOSYS M2040dn",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Paper jam J0200 on M2040dn: open rear cover, remove jammed paper carefully. Check registration roller alignment.",
        "vendor": "Kyocera", "model": "Kyocera ECOSYS M2040dn",
        "error_codes": ["J0200"], "doc_type": "service_manual",
    },
    {
        "text": "M2040dn duplex unit troubleshooting. If duplex printing fails, check the duplex feed roller and duplex path sensor. Error J0300 indicates a duplex jam.",
        "vendor": "Kyocera", "model": "Kyocera ECOSYS M2040dn",
        "error_codes": ["J0300"], "doc_type": "service_manual",
    },
    {
        "text": "M2040dn fan error E01100 — main cooling fan does not spin. Check fan connector CN501. Replace the fan motor if the fan blade is stuck.",
        "vendor": "Kyocera", "model": "Kyocera ECOSYS M2040dn",
        "error_codes": ["E01100"], "doc_type": "service_manual",
    },
    # --- HP LaserJet Pro M404n (8 chunks) ---
    {
        "text": "HP LaserJet Pro M404n error 50.2 — Fuser warm-up failure. The fuser does not reach operating temperature. Replace the fuser assembly (RM2-5399).",
        "vendor": "HP", "model": "HP LaserJet Pro M404n",
        "error_codes": ["50.2"], "doc_type": "service_manual",
    },
    {
        "text": "Error 13.A2 on M404n — Paper jam in Tray 2 pickup area. Remove tray, clear jammed paper. Inspect pickup roller for wear.",
        "vendor": "HP", "model": "HP LaserJet Pro M404n",
        "error_codes": ["13.A2"], "doc_type": "service_manual",
    },
    {
        "text": "HP M404n imaging drum replacement: Open front door, remove toner cartridge CF259A, then pull the imaging drum out. Install new drum HP CF232A.",
        "vendor": "HP", "model": "HP LaserJet Pro M404n",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Error 49.XX on M404n — firmware error / critical. Power cycle the printer. If error persists, update firmware via USB. Last resort: replace formatter board.",
        "vendor": "HP", "model": "HP LaserJet Pro M404n",
        "error_codes": ["49.XX"], "doc_type": "service_manual",
    },
    {
        "text": "HP M404n maintenance kit: fuser (200K pages), transfer roller (200K), pickup rollers (100K). Part number CF234A.",
        "vendor": "HP", "model": "HP LaserJet Pro M404n",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "HP M404n network troubleshooting: print a configuration page from the control panel. Verify IP address, subnet mask and gateway settings.",
        "vendor": "HP", "model": "HP LaserJet Pro M404n",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Error 10.10 on M404n — Supply memory error. The toner cartridge chip cannot be read. Reinstall cartridge or replace with genuine HP supply.",
        "vendor": "HP", "model": "HP LaserJet Pro M404n",
        "error_codes": ["10.10"], "doc_type": "service_manual",
    },
    {
        "text": "HP M404n duplex printing issues: check the duplex unit alignment. Error 59.F0 indicates a main motor rotation error during duplex feed.",
        "vendor": "HP", "model": "HP LaserJet Pro M404n",
        "error_codes": ["59.F0"], "doc_type": "service_manual",
    },
    # --- Ricoh MP C3003 (10 chunks) ---
    {
        "text": "Ricoh MP C3003 SC542 — Fuser thermistor open circuit. Check fuser thermistor connector and harness. Replace fuser unit if thermistor resistance is out of spec.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": ["SC542"], "doc_type": "service_manual",
    },
    {
        "text": "SC401 on MP C3003 — Transfer belt abnormality. The transfer belt home position sensor does not detect the belt. Clean or replace the transfer unit.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": ["SC401"], "doc_type": "service_manual",
    },
    {
        "text": "Ricoh MP C3003 drum replacement: open the front door, release the drum lock lever, and pull the drum unit straight out. Install new drum and reset the counter in SP mode 5810.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "SC670 — Toner density sensor error on MP C3003. Clean the TD sensor window. Perform SP mode 5-808 to recalibrate.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": ["SC670"], "doc_type": "service_manual",
    },
    {
        "text": "Ricoh MP C3003 periodic maintenance. Fuser: 160K. Drum K: 120K. Drum CMY: 80K. Transfer belt: 120K. Developer: 240K.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Scanner exposure lamp failure SC144 on MP C3003. Check lamp connector. Run SP mode 5-004 for calibration. Replace scanner unit if lamp output is degraded.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": ["SC144"], "doc_type": "service_manual",
    },
    {
        "text": "Paper jam on Ricoh MP C3003 at registration area. Open the right side door. Remove the jammed sheet. Check the registration sensor and roller pressure.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "SC302 — Charge roller error. The drum charging voltage is abnormal. Check the high-voltage power supply board and charge roller connector.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": ["SC302"], "doc_type": "service_manual",
    },
    {
        "text": "SC555 on MP C3003 — Fuser pressure roller thermistor error. Check the secondary thermistor. If resistance is abnormal, replace the fuser unit.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": ["SC555"], "doc_type": "service_manual",
    },
    {
        "text": "Ricoh MP C3003 controller board replacement. Power off, disconnect all cables. Remove 6 screws. Slide the engine board out. Transfer NVRAM chip to new board.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": [], "doc_type": "service_manual",
    },
    # --- Xerox VersaLink C405 (10 chunks) ---
    {
        "text": "Xerox VersaLink C405 error 124-211 — Fuser life reached. The fuser has exceeded its expected life. Replace the fuser unit (115R00088).",
        "vendor": "Xerox", "model": "Xerox VersaLink C405",
        "error_codes": ["124-211"], "doc_type": "service_manual",
    },
    {
        "text": "Error 010-351 on VersaLink C405 — Tray 1 paper misfeed. Check paper guides and pickup roller. Load paper correctly in the tray.",
        "vendor": "Xerox", "model": "Xerox VersaLink C405",
        "error_codes": ["010-351"], "doc_type": "service_manual",
    },
    {
        "text": "Xerox C405 drum cartridge replacement. Remove old drum cartridge by turning the orange lever. Install new drum (108R01121). Reset drum counter.",
        "vendor": "Xerox", "model": "Xerox VersaLink C405",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "VersaLink C405 wireless setup: go to Machine Status > Tools > Network > Wi-Fi. Select your SSID and enter the password. Print a configuration page to verify.",
        "vendor": "Xerox", "model": "Xerox VersaLink C405",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Xerox C405 color registration adjustment. Access Technician Menu > Image Quality > Color Registration. Run auto-registration and then manual fine-tune if needed.",
        "vendor": "Xerox", "model": "Xerox VersaLink C405",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Error 116-324 on C405 — Transfer belt life. Replace the transfer belt assembly. Part number 108R01122.",
        "vendor": "Xerox", "model": "Xerox VersaLink C405",
        "error_codes": ["116-324"], "doc_type": "service_manual",
    },
    {
        "text": "Power supply unit replacement on VersaLink C405. Disconnect AC power. Remove 4 screws on the rear panel. Slide PSU out. Install replacement PSU.",
        "vendor": "Xerox", "model": "Xerox VersaLink C405",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Error 092-651 on VersaLink C405 — Waste toner container full. Replace the waste toner bottle (108R01124). The printer will not operate until replaced.",
        "vendor": "Xerox", "model": "Xerox VersaLink C405",
        "error_codes": ["092-651"], "doc_type": "service_manual",
    },
    {
        "text": "Xerox C405 scanner ADF roller replacement. Open the ADF cover. Remove the roller retainer. Replace the ADF feed roller kit (108R01490).",
        "vendor": "Xerox", "model": "Xerox VersaLink C405",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Xerox C405 firmware update: download firmware from xerox.com. Upload via Embedded Web Server > System > Software Update. Printer restarts automatically.",
        "vendor": "Xerox", "model": "Xerox VersaLink C405",
        "error_codes": [], "doc_type": "service_manual",
    },
    # --- 5 more filler chunks to reach 50 ---
    {
        "text": "ECOSYS M2040dn laser unit cleaning: remove the LSU cover. Use compressed air to clean the laser lens. Do not touch the lens with bare fingers.",
        "vendor": "Kyocera", "model": "Kyocera ECOSYS M2040dn",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "HP M404n print density calibration: print a cleaning page from Settings > Service > Cleaning Page. Adjust density from -5 to +5.",
        "vendor": "HP", "model": "HP LaserJet Pro M404n",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "HP M404n USB firmware update procedure: download latest .rfu file. Copy to USB drive root. Insert USB, the printer auto-detects the update.",
        "vendor": "HP", "model": "HP LaserJet Pro M404n",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Ricoh MP C3003 Wi-Fi module installation: insert the wireless LAN board into the controller slot. Configure SSID via the Smart Operation Panel.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": [], "doc_type": "service_manual",
    },
    {
        "text": "Ricoh MP C3003 power supply unit. If the machine does not power on, check the AC inlet fuse. Measure DC output voltages: 3.3V, 5V, 24V. Replace PSU if voltages are out of spec.",
        "vendor": "Ricoh", "model": "Ricoh MP C3003",
        "error_codes": [], "doc_type": "service_manual",
    },
]

assert len(_CHUNKS) == 50, f"Expected 50 chunks, got {len(_CHUNKS)}"

_CHUNK_IDS = [
    str(uuid.uuid5(uuid.NAMESPACE_DNS, f"rag_eval_chunk_{i}"))
    for i in range(len(_CHUNKS))
]


# ── Eval dataset: 10 queries across 3 scenarios ──────────────────────────────


def _make_eval_dataset(tmp_dir: Path) -> Path:
    dataset = {
        "queries": [
            # --- error_code_lookup (4 queries) ---
            {
                "id": "q_ec_001",
                "query": "ошибка C6000 фьюзер не нагревается Kyocera TASKalfa",
                "expected_chunks": [_CHUNK_IDS[0]],
                "must_contain_codes": ["C6000"],
                "scenario": "error_code_lookup",
                "difficulty": "easy",
            },
            {
                "id": "q_ec_002",
                "query": "Ricoh MP C3003 ошибка SC542 термистор фьюзера",
                "expected_chunks": [_CHUNK_IDS[25]],
                "must_contain_codes": ["SC542"],
                "scenario": "error_code_lookup",
                "difficulty": "easy",
            },
            {
                "id": "q_ec_003",
                "query": "HP M404n error 50.2 fuser warm-up failure",
                "expected_chunks": [_CHUNK_IDS[17]],
                "must_contain_codes": ["50.2"],
                "scenario": "error_code_lookup",
                "difficulty": "easy",
            },
            {
                "id": "q_ec_004",
                "query": "Xerox VersaLink C405 ошибка 092-651 waste toner",
                "expected_chunks": [_CHUNK_IDS[42]],
                "must_contain_codes": ["092-651"],
                "scenario": "error_code_lookup",
                "difficulty": "easy",
            },
            # --- procedure_lookup (3 queries) ---
            {
                "id": "q_proc_001",
                "query": "как заменить девелопер на TASKalfa 3253ci процедура",
                "expected_chunks": [_CHUNK_IDS[4]],
                "scenario": "procedure_lookup",
                "difficulty": "medium",
            },
            {
                "id": "q_proc_002",
                "query": "замена драм-юнита Ricoh MP C3003 процедура",
                "expected_chunks": [_CHUNK_IDS[27]],
                "scenario": "procedure_lookup",
                "difficulty": "medium",
            },
            {
                "id": "q_proc_003",
                "query": "HP M404n imaging drum replacement procedure",
                "expected_chunks": [_CHUNK_IDS[19]],
                "scenario": "procedure_lookup",
                "difficulty": "medium",
            },
            # --- symptom_diagnosis (3 queries) ---
            {
                "id": "q_sym_001",
                "query": "принтер перегревается фьюзер выше нормы Kyocera",
                "expected_chunks": [_CHUNK_IDS[1]],
                "scenario": "symptom_diagnosis",
                "difficulty": "hard",
            },
            {
                "id": "q_sym_002",
                "query": "transfer belt sensor not detecting belt Ricoh",
                "expected_chunks": [_CHUNK_IDS[26]],
                "scenario": "symptom_diagnosis",
                "difficulty": "hard",
            },
            {
                "id": "q_sym_003",
                "query": "Xerox C405 fuser end of life needs replacement",
                "expected_chunks": [_CHUNK_IDS[35]],
                "scenario": "symptom_diagnosis",
                "difficulty": "hard",
            },
        ],
    }
    path = tmp_dir / "eval_dataset.yaml"
    path.write_text(yaml.dump(dataset, allow_unicode=True), encoding="utf-8")
    return path


# ── Availability checks ─────────────────────────────────────────────────────


def _qdrant_available() -> bool:
    try:
        c = QdrantClient(host="localhost", port=6333, prefer_grpc=False, timeout=3)
        c.get_collections()
        return True
    except Exception:
        return False


def _embedder_available() -> bool:
    try:
        from FlagEmbedding import BGEM3FlagModel  # noqa: F401

        return True
    except ImportError:
        return False


requires_qdrant = pytest.mark.skipif(
    not _qdrant_available(), reason="Qdrant not running on localhost:6333",
)
requires_embedder = pytest.mark.skipif(
    not _embedder_available(), reason="FlagEmbedding not installed",
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qdrant_rest():
    return QdrantClient(host="localhost", port=6333, prefer_grpc=False, timeout=30)


@pytest.fixture(scope="module")
def embedder():
    from config.loader import EmbeddingsConfig
    from rag.embeddings import BGEEmbedder

    cfg = EmbeddingsConfig(device="cpu", fp16=False)
    return BGEEmbedder(cfg)


@pytest.fixture(scope="module")
def rag_config():
    from config.loader import RAGConfig

    cfg = RAGConfig()
    cfg.qdrant.host = "localhost"
    cfg.qdrant.port = 6333
    cfg.qdrant.prefer_grpc = False
    return cfg


@pytest.fixture(scope="module")
def qdrant_manager(rag_config):
    from rag.qdrant_client import QdrantManager

    return QdrantManager(rag_config)


@pytest.fixture(scope="module")
def searcher(embedder, qdrant_manager, rag_config):
    from rag.search import HybridSearcher

    return HybridSearcher(qdrant_manager, embedder, rag_config)


@pytest.fixture(scope="module")
def tmp_dir() -> None:
    d = Path(tempfile.mkdtemp(prefix="rag_eval_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def eval_dataset_path(tmp_dir):
    return _make_eval_dataset(tmp_dir)


@pytest.fixture(scope="module")
def history_dir(tmp_dir):
    d = tmp_dir / "eval_history"
    d.mkdir()
    return d


@pytest.fixture(scope="module")
def seeded_collection(qdrant_rest, embedder) -> None:
    """Create collection with 50 known chunks, tear down after module."""
    qdrant_rest.create_collection(
        collection_name=_COLLECTION,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=1024, distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF),
        },
    )

    for field_name in ("vendor", "model", "error_codes", "doc_type"):
        with contextlib.suppress(Exception):
            qdrant_rest.create_payload_index(
                collection_name=_COLLECTION,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

    texts = [c["text"] for c in _CHUNKS]
    result = embedder.encode(texts, return_sparse=True)

    points = []
    for i, chunk in enumerate(_CHUNKS):
        vectors = {"dense": result.dense[i].tolist()}
        sv = result.sparse[i] if result.sparse else None
        if sv and sv.indices:
            vectors["sparse"] = qmodels.SparseVector(
                indices=sv.indices, values=sv.values,
            )

        payload = {
            "text": chunk["text"],
            "vendor": chunk["vendor"],
            "model": chunk["model"],
            "error_codes": chunk["error_codes"],
            "doc_type": chunk["doc_type"],
            "document_id": f"test_doc_{chunk['vendor'].lower()}",
        }
        points.append(
            qmodels.PointStruct(id=_CHUNK_IDS[i], vector=vectors, payload=payload),
        )

    qdrant_rest.upsert(collection_name=_COLLECTION, points=points)

    info = qdrant_rest.get_collection(_COLLECTION)
    assert info.points_count == 50, f"Expected 50, got {info.points_count}"

    yield _COLLECTION

    with contextlib.suppress(Exception):
        qdrant_rest.delete_collection(_COLLECTION)


def _make_evaluator(searcher, rag_config, eval_dataset_path, history_dir):
    from rag.evaluation import RAGEvaluator

    rag_config.evaluation.dataset_path = str(eval_dataset_path)
    rag_config.evaluation.save_history_path = str(history_dir)
    return RAGEvaluator(searcher, rag_config)


# ── Tests ────────────────────────────────────────────────────────────────────


@requires_qdrant
@requires_embedder
@pytest.mark.e2e
class TestRAGEvalGate:
    """Release gate: RAG quality must meet acceptance thresholds."""

    def test_acceptance_thresholds_pass(
        self, searcher, rag_config, eval_dataset_path, history_dir,
        seeded_collection,
    ) -> None:
        """Recall@5 >= 0.7 and MRR >= 0.5 on the fixture collection."""
        evaluator = _make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        report = evaluator.run_eval(
            seeded_collection,
            use_reranker=False,
            dataset_path=eval_dataset_path,
        )

        assert report.num_queries == 10
        assert report.all_thresholds_passed, (
            "Acceptance thresholds FAILED — release blocked.\n"
            + "\n".join(
                f"  {tc.metric}: {tc.value:.4f} vs threshold {tc.threshold} "
                f"({'PASS' if tc.passed else 'FAIL'})"
                for tc in report.threshold_checks
            )
        )

        for tc in report.threshold_checks:
            if tc.metric == "recall_at_5":
                assert tc.value >= 0.70, f"Recall@5 = {tc.value:.4f} < 0.70"
                assert tc.passed
            elif tc.metric == "mrr":
                assert tc.value >= 0.50, f"MRR = {tc.value:.4f} < 0.50"
                assert tc.passed

    def test_individual_metrics_positive(
        self, searcher, rag_config, eval_dataset_path, history_dir,
        seeded_collection,
    ) -> None:
        """All IR metrics are positive on seeded collection."""
        evaluator = _make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        report = evaluator.run_eval(
            seeded_collection,
            use_reranker=False,
            dataset_path=eval_dataset_path,
        )

        assert report.recall_at_5 > 0.0
        assert report.recall_at_10 > 0.0
        assert report.mrr > 0.0
        assert report.precision_at_5 > 0.0
        assert report.ndcg_at_10 > 0.0

    def test_per_scenario_breakdown(
        self, searcher, rag_config, eval_dataset_path, history_dir,
        seeded_collection,
    ) -> None:
        """Per-scenario metrics are computed for all 3 scenarios."""
        evaluator = _make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        report = evaluator.run_eval(
            seeded_collection,
            use_reranker=False,
            dataset_path=eval_dataset_path,
        )

        scenario_names = {s.scenario for s in report.per_scenario}
        assert scenario_names == {
            "error_code_lookup", "procedure_lookup", "symptom_diagnosis",
        }

        for s in report.per_scenario:
            assert s.num_queries > 0
            if s.scenario == "error_code_lookup":
                assert s.num_queries == 4
                assert s.recall_at_5 > 0.0
            elif s.scenario == "procedure_lookup" or s.scenario == "symptom_diagnosis":
                assert s.num_queries == 3

    def test_per_query_results_complete(
        self, searcher, rag_config, eval_dataset_path, history_dir,
        seeded_collection,
    ) -> None:
        """Every query produces a result with expected fields."""
        evaluator = _make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        report = evaluator.run_eval(
            seeded_collection,
            use_reranker=False,
            dataset_path=eval_dataset_path,
        )

        assert len(report.per_query) == 10

        for qr in report.per_query:
            assert qr.query_id
            assert qr.query
            assert qr.scenario in (
                "error_code_lookup", "procedure_lookup", "symptom_diagnosis",
            )
            assert len(qr.expected_chunks) >= 1
            assert len(qr.retrieved_chunks) >= 0
            assert 0.0 <= qr.recall_at_5 <= 1.0
            assert 0.0 <= qr.mrr <= 1.0


@requires_qdrant
@requires_embedder
@pytest.mark.e2e
class TestEvalDeterminism:
    """Eval runs are deterministic — same data, same results."""

    def test_two_runs_identical(
        self, searcher, rag_config, eval_dataset_path, history_dir,
        seeded_collection,
    ) -> None:
        evaluator = _make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        r1 = evaluator.run_eval(
            seeded_collection, use_reranker=False,
            dataset_path=eval_dataset_path,
        )
        r2 = evaluator.run_eval(
            seeded_collection, use_reranker=False,
            dataset_path=eval_dataset_path,
        )

        assert r1.recall_at_5 == r2.recall_at_5
        assert r1.mrr == r2.mrr
        assert r1.ndcg_at_10 == r2.ndcg_at_10
        assert r1.precision_at_5 == r2.precision_at_5
        assert r1.recall_at_10 == r2.recall_at_10

        for q1, q2 in zip(r1.per_query, r2.per_query, strict=True):
            assert q1.query_id == q2.query_id
            assert q1.recall_at_5 == q2.recall_at_5
            assert q1.mrr == q2.mrr


@requires_qdrant
@requires_embedder
@pytest.mark.e2e
class TestEvalHistory:
    """Save/load and delta computation."""

    def test_save_and_load_roundtrip(
        self, searcher, rag_config, eval_dataset_path, history_dir,
        seeded_collection,
    ) -> None:
        evaluator = _make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        report = evaluator.run_eval(
            seeded_collection, use_reranker=False,
            dataset_path=eval_dataset_path,
        )
        saved = evaluator.save_report(report)
        assert saved.exists()

        history = evaluator.get_history(last_n=5)
        assert len(history) >= 1

        latest = history[0]
        assert latest.recall_at_5 == report.recall_at_5
        assert latest.mrr == report.mrr
        assert latest.num_queries == report.num_queries

    def test_delta_vs_previous(
        self, searcher, rag_config, eval_dataset_path, history_dir,
        seeded_collection,
    ) -> None:
        evaluator = _make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        report = evaluator.run_eval(
            seeded_collection, use_reranker=False,
            dataset_path=eval_dataset_path,
        )

        deltas = evaluator.delta_vs_previous(report)
        assert len(deltas) == 5

        metric_names = {d.metric for d in deltas}
        assert metric_names == {
            "recall_at_5", "recall_at_10", "mrr", "precision_at_5", "ndcg_at_10",
        }

        for d in deltas:
            assert d.delta == round(d.current - d.previous, 4)


@requires_qdrant
@requires_embedder
@pytest.mark.e2e
class TestThresholdFailureDetection:
    """Verify that lowered thresholds are correctly detected as failures."""

    def test_unreachable_thresholds_fail(
        self, searcher, rag_config, eval_dataset_path, history_dir,
        seeded_collection,
    ) -> None:
        """Setting thresholds to 1.0 guarantees failure detection."""
        rag_config.evaluation.acceptance_thresholds.recall_at_5 = 1.0
        rag_config.evaluation.acceptance_thresholds.mrr = 1.0

        try:
            evaluator = _make_evaluator(
                searcher, rag_config, eval_dataset_path, history_dir,
            )

            report = evaluator.run_eval(
                seeded_collection, use_reranker=False,
                dataset_path=eval_dataset_path,
            )

            has_failure = any(not tc.passed for tc in report.threshold_checks)
            assert has_failure, (
                "Expected at least one threshold to fail with thresholds=1.0"
            )
            assert not report.all_thresholds_passed
        finally:
            rag_config.evaluation.acceptance_thresholds.recall_at_5 = 0.70
            rag_config.evaluation.acceptance_thresholds.mrr = 0.50
