"""Quick self-test (offline mode). Run: python test_run.py"""
import os

os.environ.pop("ANTHROPIC_API_KEY", None)  # force offline mode for the test
from agent import CropDoctorAgent, format_trace

CASES = [
    ("There is orange powder under my coffee leaves and many leaves are falling", "Coffee Leaf Rust"),
    ("Coffee leaves turned black and are hanging on threads after continuous rain", "Coffee Black Rot (Koleroga)"),
    ("My arabica plant is wilting, small holes with sawdust at the base of the stem", "Coffee White Stem Borer"),
    ("tiny pinhole at the tip of coffee berries and berries are dropping", "Coffee Berry Borer"),
    ("Pepper vine suddenly wilted, black spots on leaves and rotting at the base", "Pepper Foot Rot / Quick Wilt"),
    ("pepper vine slowly yellowing and thinning over months, roots have galls", "Pepper Slow Decline (Slow Wilt)"),
    ("cardamom leaves have pale streaks along the veins and clumps are stunted", "Cardamom Katte (Mosaic) Virus"),
    ("cardamom capsules rotting with foul smell after heavy rain", "Cardamom Capsule Rot (Azhukal)"),
]

passed = 0
for text, expected in CASES:
    agent = CropDoctorAgent()
    report, trace = agent.run(text)
    got = trace[0]["output"][0]["name"]
    ok = got == expected
    passed += ok
    print(("PASS" if ok else "FAIL"), "|", text[:60], "->", got)

print(f"\n{passed}/{len(CASES)} passed\n")
print("=" * 70)
agent = CropDoctorAgent()
report, trace = agent.run(CASES[0][0])
print(report)
print("=" * 70)
print(format_trace(trace))
