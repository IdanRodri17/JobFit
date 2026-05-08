# DocTor (דוקתור) — Hospital Management System

## Overview

DocTor is a hospital management system built as the final project for the SQL module of the CyberPro AI Developer Bootcamp. The project is being built step-by-step to demonstrate strong fundamentals across the full backend stack: schema design, ORM modeling, CLI tooling, and (as bonuses) a FastAPI service layer and a React frontend.

The Hebrew name דוקתור is a play on "Doctor" — fitting for a system designed for Israeli hospitals.

## Tech Stack

- **Database:** PostgreSQL (relational schema with foreign keys, constraints, and indexes)
- **ORM:** SQLAlchemy
- **CLI:** Python `argparse` (or `typer`) for administrative operations
- **API (bonus):** FastAPI with Pydantic-validated request/response models
- **Frontend (bonus):** React

## What's modeled

The system models the core entities of hospital operations:

- **Patients** with admission/discharge records
- **Doctors** with specializations and shift schedules
- **Departments** with capacity constraints
- **Appointments** linking patients, doctors, and departments
- **Medical records** tied to patient admissions

Schema design emphasizes referential integrity, sensible cascading rules (e.g. soft-deleting a department doesn't orphan its appointments), and indexed columns for common queries.

## What I'm demonstrating

- Solid relational modeling — third-normal-form schema with thoughtful denormalization where queries demand it.
- Clean SQLAlchemy ORM usage — sessions, transactions, eager vs lazy loading, alembic-style migrations.
- Practical CLI ergonomics — admin commands for adding doctors, scheduling appointments, generating reports.
- Bonus full-stack delivery — FastAPI exposing the same domain over HTTP, React frontend consuming it.

## Status

Active development, building incrementally with meaningful git commits per feature.
