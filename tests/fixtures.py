"""Fixture emails shared by the cassette seeder and the offline test tier.

These are NOT the golden dataset. The golden dataset (Phase 2) is hand-labeled
ground truth used to measure model quality; these five exist only to exercise the
harness plumbing offline. Keeping them separate stops harness fixtures from
quietly becoming eval data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fixture:
    key: str
    email: str
    category: str
    summary: str
    malformed: bool = False


FIXTURE_EMAILS: tuple[Fixture, ...] = (
    Fixture(
        key="billing_duplicate_charge",
        email=(
            "You billed my card twice this month for the Pro plan. "
            "I need the second charge reversed."
        ),
        category="billing",
        summary="Customer was billed twice for the Pro plan and wants the second charge reversed.",
    ),
    Fixture(
        key="technical_api_timeout",
        email=(
            "Our nightly export has been timing out since Tuesday. "
            "The job dies around the 40 minute mark every time."
        ),
        category="technical",
        summary=(
            "Customer reports the nightly export job timing out "
            "after about 40 minutes since Tuesday."
        ),
    ),
    Fixture(
        key="account_locked_out",
        email="I've been locked out after too many password attempts. Can you reset it?",
        category="account",
        summary="Customer is locked out after failed password attempts and requests a reset.",
    ),
    Fixture(
        key="general_partnership",
        email="We run a consultancy and would like to explore a reseller partnership.",
        category="general",
        summary="Customer wants to explore a reseller partnership opportunity.",
    ),
    # Exercises the parse_error path: the model returns prose instead of JSON.
    Fixture(
        key="malformed_output",
        email="???",
        category="general",
        summary="",
        malformed=True,
    ),
)


def by_key(key: str) -> Fixture:
    for fixture in FIXTURE_EMAILS:
        if fixture.key == key:
            return fixture
    raise KeyError(key)
