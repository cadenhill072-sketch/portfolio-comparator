-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New query)

-- Saved portfolios table
create table if not exists saved_portfolios (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  name text not null,
  weights jsonb not null,
  created_at timestamptz default now(),
  unique(user_id, name)
);

-- Row-level security: users can only see/edit their own portfolios
alter table saved_portfolios enable row level security;

create policy "Users can read own portfolios"
  on saved_portfolios for select
  using (auth.uid() = user_id);

create policy "Users can insert own portfolios"
  on saved_portfolios for insert
  with check (auth.uid() = user_id);

create policy "Users can update own portfolios"
  on saved_portfolios for update
  using (auth.uid() = user_id);

create policy "Users can delete own portfolios"
  on saved_portfolios for delete
  using (auth.uid() = user_id);

-- Alert preferences table (for email alerts, phase 2)
create table if not exists alert_preferences (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null unique,
  sharpe_threshold numeric default 0.5,
  alert_enabled boolean default false,
  created_at timestamptz default now()
);

alter table alert_preferences enable row level security;

create policy "Users can manage own alerts"
  on alert_preferences for all
  using (auth.uid() = user_id);
