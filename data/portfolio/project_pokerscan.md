# PokerScan / PokerVision — YOLOv8 Card Detection

## Overview

PokerScan (also called PokerVision) is a computer vision application that detects and classifies playing cards in real time using a custom-trained YOLOv8 model. Built as a portfolio project to demonstrate end-to-end ML deployment skills — from dataset curation through model training to a deployed full-stack web application.

## Tech Stack

- **ML model:** YOLOv8 (Ultralytics), custom-trained on a labeled card dataset
- **Backend:** FastAPI serving model inference over HTTP
- **Frontend:** React for the user interface
- **Deployment:** Hugging Face Spaces (model demo) and Netlify (frontend)

## What I built

- Curated and labeled a card detection dataset.
- Trained a YOLOv8 detector to recognize all 52 standard playing cards with high accuracy.
- Wrapped the trained model in a FastAPI service that accepts an uploaded image and returns the detected cards with bounding boxes and confidence scores.
- Built a React frontend that lets users upload or capture an image and see the detected cards visualized.
- Deployed the full system to Hugging Face Spaces and Netlify so anyone can try it without local setup.

## Why this project mattered

PokerScan was my first end-to-end ML deployment project: it forced me to handle the full lifecycle of an ML product (data → training → API → frontend → deployment), not just the modeling step. The specific skills it built — dataset curation, training loops, serving inference behind a clean API, and deploying public ML demos — translate directly to production AI engineering work.
