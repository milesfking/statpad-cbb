export type SeasonStatus = "live" | "final";

export interface SeasonMeta {
  season: number;
  label: string;
  status: SeasonStatus;
  as_of?: string;
}

export interface RankingsMeta {
  generated_at: string;
  default_season: number;
  seasons: SeasonMeta[];
}

export type TabId = "teams" | "conferences" | "players";

export interface TeamRow {
  rank: number;
  off_rank: number;
  def_rank: number;
  team_id: number;
  school: string;
  conference: string;
  off_rating: number;
  def_rating: number;
  net_rating: number;
}

export interface ConferenceRow {
  rank: number;
  off_rank: number;
  def_rank: number;
  conference: string;
  off_rating: number;
  def_rating: number;
  net_rating: number;
}

export interface PlayerRow {
  rank: number;
  off_rank: number;
  def_rank: number;
  player_id: number;
  name?: string;
  team_id?: number;
  school?: string;
  conference?: string;
  minutes?: number;
  off_rating: number;
  def_rating: number;
  net_rating: number;
}
