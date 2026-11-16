# Design decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-17 | `make ui` serves MLflow on port **5001** instead of 5000 | macOS AirPlay Receiver (ControlCenter) owns port 5000 and answers 403. |
| 2026-11-17 | Default DVC remote = local folder `../synthguard-dvc-remote` (outside repo) | Zero credentials; graders rebuild from the public URL anyway. Shared remote (Drive/DagsHub) optional later. |
