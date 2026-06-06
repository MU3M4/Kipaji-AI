# policies.py
from google.antigravity.hooks.policy import deny, allow

# Declarative policy chain: Block all dangerous OS commands, allow read-only lookups
KIPAJI_SAFETY_POLICIES = [
    deny("run_command", when=lambda args: any(cmd in args.get("CommandLine", "") for cmd in ["rm", "sudo", "chmod", "wget", "curl"])),
    allow("view_file"), # Allow the agent to safely read ledger files
    deny("*") # Explicit block on any other unapproved system-level capability
]