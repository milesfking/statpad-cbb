import type { Column } from "./components/SortableTable";
import { displayConference } from "./format";
import type { ConferenceRow, PlayerRow, TeamRow } from "./types";

export const teamColumns: Column<TeamRow>[] = [
  {
    id: "rank",
    label: "Rank",
    align: "right",
    getValue: (r) => r.rank,
  },
  {
    id: "school",
    label: "Team",
    getValue: (r) => r.school,
  },
  {
    id: "conference",
    label: "Conference",
    getValue: (r) => r.conference,
    render: (r) => displayConference(r.conference),
  },
  {
    id: "off_rank",
    label: "Off Rank",
    align: "right",
    getValue: (r) => r.off_rank,
  },
  {
    id: "off_rating",
    label: "Off",
    align: "right",
    getValue: (r) => r.off_rating,
  },
  {
    id: "def_rank",
    label: "Def Rank",
    align: "right",
    getValue: (r) => r.def_rank,
  },
  {
    id: "def_rating",
    label: "Def",
    align: "right",
    getValue: (r) => r.def_rating,
  },
  {
    id: "net_rating",
    label: "Net Rtg",
    align: "right",
    getValue: (r) => r.net_rating,
  },
];

export const conferenceColumns: Column<ConferenceRow>[] = [
  {
    id: "rank",
    label: "Rank",
    align: "right",
    getValue: (r) => r.rank,
  },
  {
    id: "conference",
    label: "Conference",
    getValue: (r) => r.conference,
    render: (r) => displayConference(r.conference),
  },
  {
    id: "off_rank",
    label: "Off Rank",
    align: "right",
    getValue: (r) => r.off_rank,
  },
  {
    id: "off_rating",
    label: "Off",
    align: "right",
    getValue: (r) => r.off_rating,
  },
  {
    id: "def_rank",
    label: "Def Rank",
    align: "right",
    getValue: (r) => r.def_rank,
  },
  {
    id: "def_rating",
    label: "Def",
    align: "right",
    getValue: (r) => r.def_rating,
  },
  {
    id: "net_rating",
    label: "Net Rtg",
    align: "right",
    getValue: (r) => r.net_rating,
  },
];

export const playerColumns: Column<PlayerRow>[] = [
  {
    id: "rank",
    label: "Rank",
    align: "right",
    getValue: (r) => r.rank,
  },
  {
    id: "name",
    label: "Player",
    getValue: (r) => r.name ?? `Player ${r.player_id}`,
  },
  {
    id: "school",
    label: "Team",
    getValue: (r) => r.school,
  },
  {
    id: "conference",
    label: "Conference",
    getValue: (r) => r.conference,
    render: (r) => displayConference(r.conference),
  },
  {
    id: "minutes",
    label: "Min",
    align: "right",
    getValue: (r) => r.minutes,
  },
  {
    id: "off_rank",
    label: "Off Rank",
    align: "right",
    getValue: (r) => r.off_rank,
  },
  {
    id: "off_rating",
    label: "Off",
    align: "right",
    getValue: (r) => r.off_rating,
  },
  {
    id: "def_rank",
    label: "Def Rank",
    align: "right",
    getValue: (r) => r.def_rank,
  },
  {
    id: "def_rating",
    label: "Def",
    align: "right",
    getValue: (r) => r.def_rating,
  },
  {
    id: "net_rating",
    label: "Net Rtg",
    align: "right",
    getValue: (r) => r.net_rating,
  },
];
