"""
agent/tools.py
Mock tool functions executed by the agent during lead capture.
"""


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulates sending a qualified lead to a CRM / marketing backend.

    In production this would make an HTTP POST to your CRM API.
    For now it simply prints the captured data and returns a success string.

    Args:
        name     : Full name of the prospect.
        email    : Email address of the prospect.
        platform : Creator platform (e.g. YouTube, Instagram, TikTok).

    Returns:
        A confirmation string.
    """
    print(f"\n{'='*60}")
    print(f"  ✅  Lead captured successfully!")
    print(f"  Name    : {name}")
    print(f"  Email   : {email}")
    print(f"  Platform: {platform}")
    print(f"{'='*60}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"
