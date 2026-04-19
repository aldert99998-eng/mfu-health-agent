"""E2E verification for hybrid search + reranker (Phase 4.4).

Seeds a temporary Qdrant collection with ~50 diverse chunks, then checks:
1. Known error-code query lands in top-3.
2. Model filter restricts results to that model only.
3. Reranker changes ordering vs plain hybrid.

Requires: running Qdrant on localhost:6333 and BGE-M3 model cached.
Run:  python -m pytest tests/e2e/test_hybrid_search.py -v -s
"""

from __future__ import annotations

import contextlib
import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ── Fixture data: 50 synthetic chunks across 3 vendors / 5 models ──────────

_COLLECTION = f"_test_search_{uuid.uuid4().hex[:8]}"

CHUNKS: list[dict] = []

_KYOCERA_3253_CHUNKS = [
    {
        "text": "Error Code C6000 — Abnormal fuser heater temperature. The fuser unit fails to reach target temperature within 30 seconds. Replace the fuser thermistor or the fuser heater lamp.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["C6000"],
        "doc_type": "service_manual",
        "section": "7.2 Fuser Unit Troubleshooting",
    },
    {
        "text": "Error Code C6020 — Fuser overheat. The fuser temperature exceeds the upper safety limit. Check thermistor wiring and fuser unit connector CN201.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["C6020"],
        "doc_type": "service_manual",
        "section": "7.2 Fuser Unit Troubleshooting",
    },
    {
        "text": "Error Code C0800 — Image density sensor malfunction. Clean the ID sensor lens. If the error persists, replace the ID sensor assembly.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["C0800"],
        "doc_type": "service_manual",
        "section": "8.1 Image Quality",
    },
    {
        "text": "The drum unit (OPC drum) should be replaced every 200,000 pages. Check drum surface for scratches or coating damage.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "8.1 Drum Unit Maintenance",
    },
    {
        "text": "Developer unit replacement procedure: 1) Open front cover. 2) Remove developer unit by pulling the green handle. 3) Install the new developer unit.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "8.3 Developer Replacement",
    },
    {
        "text": "Paper feed roller replacement: Remove cassette tray. Release the roller retainer clips. Replace the pickup roller and feed roller set.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "9.1 Paper Feed",
    },
    {
        "text": "Periodic maintenance table for TASKalfa 3253ci. Fuser replacement: 600K pages. Drum: 200K. Developer: 600K. Transfer belt: 300K.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "10. PM Schedule",
    },
    {
        "text": "Scanner CIS unit calibration for TASKalfa 3253ci. Run auto calibration from service mode 8001. Replace CIS if streaks persist.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "11.2 Scanner Calibration",
    },
    {
        "text": "Error Code J0510 — Paper jam at the transfer unit area. Remove jammed paper from the transfer belt section. Check transfer belt tension.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["J0510"],
        "doc_type": "service_manual",
        "section": "6.1 Jam Codes",
    },
    {
        "text": "Error Code E00010 — Main motor failure. Check motor connector CN101 and PCB relay. Replace main motor assembly if drive signal is absent.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["E00010"],
        "doc_type": "service_manual",
        "section": "5.3 Motor Errors",
    },
]

_KYOCERA_M2040_CHUNKS = [
    {
        "text": "Error Code C6000 on ECOSYS M2040dn — Fuser heater error. This model uses a thin-film fuser. Replace the fuser unit assembly (FK-1150).",
        "vendor": "Kyocera",
        "model": "Kyocera ECOSYS M2040dn",
        "error_codes": ["C6000"],
        "doc_type": "service_manual",
        "section": "7.1 Fuser Troubleshooting",
    },
    {
        "text": "ECOSYS M2040dn — drum replacement procedure. Use DK-1150 drum kit. Expected yield: 100,000 pages.",
        "vendor": "Kyocera",
        "model": "Kyocera ECOSYS M2040dn",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "8.1 Drum Replacement",
    },
    {
        "text": "Toner supply error on M2040dn. Error code C7990 — toner sensor detects empty cartridge. Replace TK-1170 toner cartridge.",
        "vendor": "Kyocera",
        "model": "Kyocera ECOSYS M2040dn",
        "error_codes": ["C7990"],
        "doc_type": "service_manual",
        "section": "8.2 Toner System",
    },
    {
        "text": "M2040dn network configuration: access embedded web server via https://<ip>. Default admin password: Admin00. Configure SMTP for scan-to-email.",
        "vendor": "Kyocera",
        "model": "Kyocera ECOSYS M2040dn",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "12.1 Network Setup",
    },
    {
        "text": "Paper jam J0200 on M2040dn: open rear cover, remove jammed paper carefully. Check registration roller alignment.",
        "vendor": "Kyocera",
        "model": "Kyocera ECOSYS M2040dn",
        "error_codes": ["J0200"],
        "doc_type": "service_manual",
        "section": "6.2 Jam Codes",
    },
]

_HP_M404_CHUNKS = [
    {
        "text": "HP LaserJet Pro M404n error 50.2 — Fuser warm-up failure. The fuser does not reach operating temperature. Replace the fuser assembly (RM2-5399).",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": ["50.2"],
        "doc_type": "service_manual",
        "section": "Chapter 8 — Fuser Errors",
    },
    {
        "text": "Error 13.A2 on M404n — Paper jam in Tray 2 pickup area. Remove tray, clear jammed paper. Inspect pickup roller for wear.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": ["13.A2"],
        "doc_type": "service_manual",
        "section": "Chapter 6 — Jams",
    },
    {
        "text": "HP M404n imaging drum replacement: Open front door, remove toner cartridge CF259A, then pull the imaging drum out. Install new drum HP CF232A.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Chapter 9 — Consumables",
    },
    {
        "text": "Error 49.XX on M404n — firmware error / critical. Power cycle the printer. If error persists, update firmware via USB. Last resort: replace formatter board.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": ["49.XX"],
        "doc_type": "service_manual",
        "section": "Chapter 10 — System Errors",
    },
    {
        "text": "HP M404n maintenance kit: fuser (200K pages), transfer roller (200K), pickup rollers (100K). Part number CF234A.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Chapter 11 — PM Kit",
    },
    {
        "text": "HP M404n network troubleshooting: print a configuration page from the control panel. Verify IP address, subnet mask and gateway settings.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Chapter 12 — Network",
    },
    {
        "text": "Error 10.10 on M404n — Supply memory error. The toner cartridge chip cannot be read. Reinstall cartridge or replace with genuine HP supply.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": ["10.10"],
        "doc_type": "service_manual",
        "section": "Chapter 8 — Supply Errors",
    },
]

_RICOH_C3003_CHUNKS = [
    {
        "text": "Ricoh MP C3003 SC542 — Fuser thermistor open circuit. Check fuser thermistor connector and harness. Replace fuser unit if thermistor resistance is out of spec.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": ["SC542"],
        "doc_type": "service_manual",
        "section": "Section 7: Fuser",
    },
    {
        "text": "SC401 on MP C3003 — Transfer belt abnormality. The transfer belt home position sensor does not detect the belt. Clean or replace the transfer unit.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": ["SC401"],
        "doc_type": "service_manual",
        "section": "Section 8: Transfer",
    },
    {
        "text": "Ricoh MP C3003 drum replacement: open the front door, release the drum lock lever, and pull the drum unit straight out. Install new drum and reset the counter in SP mode 5810.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 9: Drum",
    },
    {
        "text": "SC670 — Toner density sensor error on MP C3003. Clean the TD sensor window. Perform SP mode 5-808 to recalibrate.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": ["SC670"],
        "doc_type": "service_manual",
        "section": "Section 10: Toner",
    },
    {
        "text": "Ricoh MP C3003 periodic maintenance. Fuser: 160K. Drum K: 120K. Drum CMY: 80K. Transfer belt: 120K. Developer: 240K.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 15: PM Table",
    },
    {
        "text": "Scanner exposure lamp failure SC144 on MP C3003. Check lamp connector. Run SP mode 5-004 for calibration. Replace scanner unit if lamp output is degraded.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": ["SC144"],
        "doc_type": "service_manual",
        "section": "Section 12: Scanner",
    },
    {
        "text": "Paper jam on Ricoh MP C3003 at registration area. Open the right side door. Remove the jammed sheet. Check the registration sensor and roller pressure.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 6: Jams",
    },
    {
        "text": "SC302 — Charge roller error. The drum charging voltage is abnormal. Check the high-voltage power supply board and charge roller connector.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": ["SC302"],
        "doc_type": "service_manual",
        "section": "Section 7: Charging",
    },
]

_XEROX_C405_CHUNKS = [
    {
        "text": "Xerox VersaLink C405 error 124-211 — Fuser life reached. The fuser has exceeded its expected life. Replace the fuser unit (115R00088).",
        "vendor": "Xerox",
        "model": "Xerox VersaLink C405",
        "error_codes": ["124-211"],
        "doc_type": "service_manual",
        "section": "Section 7: Fuser",
    },
    {
        "text": "Error 010-351 on VersaLink C405 — Tray 1 paper misfeed. Check paper guides and pickup roller. Load paper correctly in the tray.",
        "vendor": "Xerox",
        "model": "Xerox VersaLink C405",
        "error_codes": ["010-351"],
        "doc_type": "service_manual",
        "section": "Section 6: Jams",
    },
    {
        "text": "Xerox C405 drum cartridge replacement. Remove old drum cartridge by turning the orange lever. Install new drum (108R01121). Reset drum counter.",
        "vendor": "Xerox",
        "model": "Xerox VersaLink C405",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 9: Consumables",
    },
    {
        "text": "VersaLink C405 wireless setup: go to Machine Status > Tools > Network > Wi-Fi. Select your SSID and enter the password. Print a configuration page to verify.",
        "vendor": "Xerox",
        "model": "Xerox VersaLink C405",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 13: Network",
    },
    {
        "text": "Xerox C405 color registration adjustment. Access Technician Menu > Image Quality > Color Registration. Run auto-registration and then manual fine-tune if needed.",
        "vendor": "Xerox",
        "model": "Xerox VersaLink C405",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 11: Image Quality",
    },
    {
        "text": "Error 116-324 on C405 — Transfer belt life. Replace the transfer belt assembly. Part number 108R01122.",
        "vendor": "Xerox",
        "model": "Xerox VersaLink C405",
        "error_codes": ["116-324"],
        "doc_type": "service_manual",
        "section": "Section 8: Transfer",
    },
    {
        "text": "Power supply unit replacement on VersaLink C405. Disconnect AC power. Remove 4 screws on the rear panel. Slide PSU out. Install replacement PSU.",
        "vendor": "Xerox",
        "model": "Xerox VersaLink C405",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 14: Power Supply",
    },
    {
        "text": "Error 092-651 on VersaLink C405 — Waste toner container full. Replace the waste toner bottle (108R01124). The printer will not operate until replaced.",
        "vendor": "Xerox",
        "model": "Xerox VersaLink C405",
        "error_codes": ["092-651"],
        "doc_type": "service_manual",
        "section": "Section 10: Toner System",
    },
    {
        "text": "Xerox C405 scanner ADF roller replacement. Open the ADF cover. Remove the roller retainer. Replace the ADF feed roller kit (108R01490).",
        "vendor": "Xerox",
        "model": "Xerox VersaLink C405",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 12: Scanner/ADF",
    },
    {
        "text": "Xerox C405 firmware update: download firmware from xerox.com. Upload via Embedded Web Server > System > Software Update. Printer restarts automatically.",
        "vendor": "Xerox",
        "model": "Xerox VersaLink C405",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 15: Firmware",
    },
]

_EXTRA_CHUNKS = [
    {
        "text": "M2040dn duplex unit troubleshooting. If duplex printing fails, check the duplex feed roller and duplex path sensor. Error J0300 indicates a duplex jam.",
        "vendor": "Kyocera",
        "model": "Kyocera ECOSYS M2040dn",
        "error_codes": ["J0300"],
        "doc_type": "service_manual",
        "section": "6.3 Duplex Jams",
    },
    {
        "text": "ECOSYS M2040dn laser unit cleaning: remove the LSU cover. Use compressed air to clean the laser lens. Do not touch the lens with bare fingers.",
        "vendor": "Kyocera",
        "model": "Kyocera ECOSYS M2040dn",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "11.1 Laser Unit",
    },
    {
        "text": "M2040dn fan error E01100 — main cooling fan does not spin. Check fan connector CN501. Replace the fan motor if the fan blade is stuck.",
        "vendor": "Kyocera",
        "model": "Kyocera ECOSYS M2040dn",
        "error_codes": ["E01100"],
        "doc_type": "service_manual",
        "section": "5.4 Fan Errors",
    },
    {
        "text": "HP M404n duplex printing issues: check the duplex unit alignment. Error 59.F0 indicates a main motor rotation error during duplex feed.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": ["59.F0"],
        "doc_type": "service_manual",
        "section": "Chapter 7 — Duplex",
    },
    {
        "text": "HP M404n print density calibration: print a cleaning page from Settings > Service > Cleaning Page. Adjust density from -5 to +5.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Chapter 9 — Image Quality",
    },
    {
        "text": "HP M404n USB firmware update procedure: download latest .rfu file. Copy to USB drive root. Insert USB, the printer auto-detects the update.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Chapter 13 — Firmware",
    },
    {
        "text": "Ricoh MP C3003 controller board replacement. Power off, disconnect all cables. Remove 6 screws. Slide the engine board out. Transfer NVRAM chip to new board.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 14: Controller",
    },
    {
        "text": "SC555 on MP C3003 — Fuser pressure roller thermistor error. Check the secondary thermistor. If resistance is abnormal, replace the fuser unit.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": ["SC555"],
        "doc_type": "service_manual",
        "section": "Section 7: Fuser",
    },
    {
        "text": "Ricoh MP C3003 Wi-Fi module installation: insert the wireless LAN board into the controller slot. Configure SSID via the Smart Operation Panel.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 16: Network",
    },
    {
        "text": "Ricoh MP C3003 power supply unit. If the machine does not power on, check the AC inlet fuse. Measure DC output voltages: 3.3V, 5V, 24V. Replace PSU if voltages are out of spec.",
        "vendor": "Ricoh",
        "model": "Ricoh MP C3003",
        "error_codes": [],
        "doc_type": "service_manual",
        "section": "Section 13: Power Supply",
    },
]

CHUNKS.extend(_KYOCERA_3253_CHUNKS)
CHUNKS.extend(_KYOCERA_M2040_CHUNKS)
CHUNKS.extend(_HP_M404_CHUNKS)
CHUNKS.extend(_RICOH_C3003_CHUNKS)
CHUNKS.extend(_XEROX_C405_CHUNKS)
CHUNKS.extend(_EXTRA_CHUNKS)

assert len(CHUNKS) == 50, f"Expected 50 chunks, got {len(CHUNKS)}"


# ── Fixtures ───────────────────────────────────────────────────────────────


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
def seeded_collection(qdrant_rest, embedder) -> None:
    """Create a temp collection with 50 chunks, tear down after tests."""
    dense_size = 1024

    qdrant_rest.create_collection(
        collection_name=_COLLECTION,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=dense_size, distance=qmodels.Distance.COSINE,
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

    texts = [c["text"] for c in CHUNKS]
    result = embedder.encode(texts, return_sparse=True)

    points = []
    for i, chunk in enumerate(CHUNKS):
        vectors = {"dense": result.dense[i].tolist()}
        sv = result.sparse[i] if result.sparse else None
        if sv and sv.indices:
            vectors["sparse"] = qmodels.SparseVector(
                indices=sv.indices, values=sv.values,
            )

        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"test_chunk_{i}"))
        payload = {
            "text": chunk["text"],
            "vendor": chunk["vendor"],
            "model": chunk["model"],
            "error_codes": chunk["error_codes"],
            "doc_type": chunk["doc_type"],
            "section": chunk["section"],
            "document_id": f"test_doc_{chunk['vendor'].lower()}",
        }
        points.append(qmodels.PointStruct(id=chunk_id, vector=vectors, payload=payload))

    qdrant_rest.upsert(collection_name=_COLLECTION, points=points)

    info = qdrant_rest.get_collection(_COLLECTION)
    assert info.points_count == 50, f"Expected 50, got {info.points_count}"

    yield _COLLECTION

    qdrant_rest.delete_collection(_COLLECTION)


@pytest.fixture(scope="module")
def searcher(embedder, seeded_collection):
    """HybridSearcher without reranker."""
    from config.loader import RAGConfig
    from rag.qdrant_client import QdrantManager

    cfg = RAGConfig()
    cfg.qdrant.host = "localhost"
    cfg.qdrant.port = 6333
    cfg.qdrant.prefer_grpc = False

    mgr = QdrantManager(cfg)

    from rag.search import HybridSearcher
    return HybridSearcher(mgr, embedder, cfg)


# ── Tests ──────────────────────────────────────────────────────────────────


@requires_qdrant
@requires_embedder
class TestHybridSearch:
    """Phase 4.4 verification suite."""

    def test_known_error_code_in_top3(self, searcher, seeded_collection) -> None:
        """Query for a known error code finds the matching chunk in top-3."""
        results = searcher.search(
            "Ошибка C6000 фьюзер не нагревается",
            seeded_collection,
            top_k=10,
            use_reranker=False,
        )

        assert len(results) > 0, "No results returned"

        top3_texts = [r.text for r in results[:3]]
        top3_codes = []
        for r in results[:3]:
            codes = r.payload.get("error_codes", [])
            top3_codes.extend(codes)

        assert "C6000" in top3_codes, (
            f"C6000 not found in top-3 error codes: {top3_codes}\n"
            f"Top-3 texts:\n" + "\n---\n".join(top3_texts)
        )

    def test_model_filter_restricts_results(self, searcher, seeded_collection) -> None:
        """Filter by model returns only chunks for that model."""
        results = searcher.search(
            "fuser error troubleshooting",
            seeded_collection,
            top_k=20,
            filters={"model": "Kyocera TASKalfa 3253ci"},
            use_reranker=False,
        )

        assert len(results) > 0, "No results with model filter"

        for r in results:
            assert r.payload.get("model") == "Kyocera TASKalfa 3253ci", (
                f"Result has wrong model: {r.payload.get('model')}"
            )

    def test_model_alias_filter(self, searcher, seeded_collection) -> None:
        """Filter with alias 'm2040dn' resolves to canonical model name."""
        results = searcher.search(
            "drum replacement",
            seeded_collection,
            top_k=10,
            filters={"model": "m2040dn"},
            use_reranker=False,
        )

        assert len(results) > 0, "No results with alias filter 'm2040dn'"

        for r in results:
            assert r.payload.get("model") == "Kyocera ECOSYS M2040dn", (
                f"Alias filter resolved to wrong model: {r.payload.get('model')}"
            )

    def test_vendor_filter(self, searcher, seeded_collection) -> None:
        """Filter by vendor returns only that vendor's chunks."""
        results = searcher.search(
            "error code troubleshooting",
            seeded_collection,
            top_k=20,
            filters={"vendor": "HP"},
            use_reranker=False,
        )

        assert len(results) > 0, "No results with vendor=HP filter"

        for r in results:
            assert r.payload.get("vendor") == "HP", (
                f"Result has wrong vendor: {r.payload.get('vendor')}"
            )

    def test_multi_vendor_search(self, searcher, seeded_collection) -> None:
        """Unfiltered search returns results from multiple vendors."""
        results = searcher.search(
            "fuser replacement procedure",
            seeded_collection,
            top_k=10,
            use_reranker=False,
        )

        vendors = {r.payload.get("vendor") for r in results}
        assert len(vendors) >= 2, (
            f"Expected multiple vendors, got: {vendors}"
        )


@requires_qdrant
@requires_embedder
class TestReranker:
    """Reranker verification — ordering changes after cross-encoder."""

    @pytest.fixture(scope="class")
    def reranker(self):
        try:
            from config.loader import RerankerConfig
            from rag.reranker import BGEReranker
            cfg = RerankerConfig(device="cpu")
            return BGEReranker(cfg)
        except Exception as e:
            pytest.skip(f"Reranker not available: {e}")

    @pytest.fixture(scope="class")
    def searcher_with_reranker(self, embedder, reranker):
        from config.loader import RAGConfig
        from rag.qdrant_client import QdrantManager
        from rag.search import HybridSearcher

        cfg = RAGConfig()
        cfg.qdrant.host = "localhost"
        cfg.qdrant.port = 6333
        cfg.qdrant.prefer_grpc = False

        mgr = QdrantManager(cfg)
        return HybridSearcher(mgr, embedder, cfg, reranker=reranker)

    def test_reranker_changes_order(
        self, searcher, searcher_with_reranker, seeded_collection,
    ) -> None:
        """Reranker produces different top-8 ordering than plain hybrid."""
        query = "как заменить фьюзер на Kyocera"

        without = searcher.search(
            query, seeded_collection, top_k=8, use_reranker=False,
        )
        with_rr = searcher_with_reranker.search(
            query, seeded_collection, top_k=8, use_reranker=True,
        )

        ids_without = [r.chunk_id for r in without]
        ids_with = [r.chunk_id for r in with_rr]

        assert ids_without != ids_with, (
            "Reranker did not change the ordering at all — "
            "expected different top-8 ordering.\n"
            f"Without: {ids_without}\n"
            f"With:    {ids_with}"
        )
