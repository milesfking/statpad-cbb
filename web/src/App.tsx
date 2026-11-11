import { useCallback, useEffect, useState } from "react";
import { SortableTable } from "./components/SortableTable";
import {
  conferenceColumns,
  playerColumns,
  teamColumns,
} from "./columns";
import { formatAsOfLine, formatSiteLastUpdated } from "./format";
import type {
  ConferenceRow,
  PlayerRow,
  RankingsMeta,
  TabId,
  TeamRow,
} from "./types";

const DATA_ROOT = import.meta.env.BASE_URL + "data/";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to load ${path} (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export default function App() {
  const [meta, setMeta] = useState<RankingsMeta | null>(null);
  const [season, setSeason] = useState<number | null>(null);
  const [tab, setTab] = useState<TabId>("teams");
  const [teams, setTeams] = useState<TeamRow[]>([]);
  const [conferences, setConferences] = useState<ConferenceRow[]>([]);
  const [players, setPlayers] = useState<PlayerRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadSeason = useCallback(async (year: number) => {
    setLoading(true);
    setError(null);
    try {
      const prefix = `${DATA_ROOT}${year}/`;
      const [t, c, p] = await Promise.all([
        fetchJson<TeamRow[]>(`${prefix}teams.json`),
        fetchJson<ConferenceRow[]>(`${prefix}conferences.json`),
        fetchJson<PlayerRow[]>(`${prefix}players.json`),
      ]);
      setTeams(t);
      setConferences(c);
      setPlayers(p);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Load failed");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const m = await fetchJson<RankingsMeta>(`${DATA_ROOT}rankings_meta.json`);
        if (cancelled) return;
        setMeta(m);
        const initial = m.default_season;
        setSeason(initial);
        await loadSeason(initial);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Load failed");
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loadSeason]);

  const seasonMeta = meta?.seasons.find((s) => s.season === season);

  return (
    <div className="page">
      <header className="header">
        <h1>Statpad CBB</h1>
        <div className="header-controls">
          {meta && (
            <label className="season-select">
              <span className="sr-only">Season</span>
              <select
                value={season ?? ""}
                onChange={(e) => {
                  const y = Number(e.target.value);
                  setSeason(y);
                  void loadSeason(y);
                }}
              >
                {meta.seasons.map((s) => (
                  <option key={s.season} value={s.season}>
                    {s.label}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      </header>

      {seasonMeta?.as_of && (
        <p className="as-of">{formatAsOfLine(seasonMeta.as_of)}</p>
      )}

      <nav className="tabs" aria-label="Rankings views">
        {(
          [
            ["teams", "Teams"],
            ["conferences", "Conferences"],
            ["players", "Players"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            className={tab === id ? "tab active" : "tab"}
            aria-current={tab === id ? "page" : undefined}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      {error && <p className="error">{error}</p>}
      {loading && <p className="loading">Loading…</p>}

      {!loading && !error && tab === "teams" && (
        <SortableTable
          rows={teams}
          columns={teamColumns}
          defaultSortId="rank"
          defaultDir="asc"
          rowKey={(r) => r.team_id}
        />
      )}
      {!loading && !error && tab === "conferences" && (
        <SortableTable
          rows={conferences}
          columns={conferenceColumns}
          defaultSortId="rank"
          defaultDir="asc"
          rowKey={(r) => r.conference}
        />
      )}
      {!loading && !error && tab === "players" && (
        <SortableTable
          rows={players}
          columns={playerColumns}
          defaultSortId="rank"
          defaultDir="asc"
          rowKey={(r) => r.player_id}
        />
      )}

      {meta && (
        <p className="site-updated">{formatSiteLastUpdated(meta.generated_at)}</p>
      )}
    </div>
  );
}
