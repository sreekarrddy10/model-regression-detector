-- Run history. SQLite keeps this inspectable with no server and no migrations
-- tooling; the eval suite's own history should never be harder to query than
-- the thing it measures.

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    git_sha          TEXT NOT NULL,
    prompt_version   TEXT NOT NULL,
    dataset_version  TEXT NOT NULL,
    dataset_hash     TEXT NOT NULL,
    model            TEXT NOT NULL,
    judge_model      TEXT,
    repeats          INTEGER NOT NULL,
    tier             TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    judge_kappa      REAL
);

CREATE TABLE IF NOT EXISTS case_results (
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    case_id         TEXT NOT NULL,
    repeat_idx      INTEGER NOT NULL,
    raw_output      TEXT NOT NULL,
    category        TEXT,
    summary         TEXT,
    parse_error     TEXT,
    schema_valid    INTEGER NOT NULL,
    category_match  INTEGER NOT NULL,
    judge_score     INTEGER,
    judge_rationale TEXT,
    latency_ms      INTEGER NOT NULL,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    cost_usd        REAL,
    error           TEXT,
    PRIMARY KEY (run_id, case_id, repeat_idx)
);

-- Indexed on dataset_hash because comparability is the first question asked of
-- any historical run: a baseline scored against different ground truth is not a
-- baseline at all.
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset_hash, started_at);
CREATE INDEX IF NOT EXISTS idx_runs_prompt ON runs(prompt_version, started_at);
CREATE INDEX IF NOT EXISTS idx_case_results_case ON case_results(case_id);
