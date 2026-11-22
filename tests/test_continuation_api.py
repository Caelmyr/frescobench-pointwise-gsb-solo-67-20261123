"""End-to-end continuation flow over the HTTP API.

Run from the project root: python3 tests/test_continuation_api.py
"""
import pathlib
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from frescobench.web import create_app


OPTIONS = {
    "sensitivity": .63, "spot_radius": 5, "polarity": "both", "minimum_area": 6, "maximum_area": 330,
}


class ContinuationFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp(prefix="frescobench-continuation-")
        cls.app = create_app(cls.directory)
        cls.client = cls.app.test_client()
        cls.client.post("/api/training")
        items = cls.client.get("/api/specimens").get_json()["items"]
        cls.tile = next(item for item in items if item["name"].startswith("Ultramarine"))["id"]
        tile = cls.client.get(f"/api/specimens/{cls.tile}").get_json()
        cls.photos = {photo["role"]: photo["id"] for photo in tile["photos"]}
        cls.region = next(region for region in tile["regions"] if region["name"] == "Painted face")["id"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def run_defects(self, role):
        response = self.client.post(
            f"/api/specimens/{self.tile}/results",
            json={
                "kind": "defects", "photo_id": self.photos[role], "region_id": self.region,
                "apply_neutral": True, "options": OPTIONS,
            },
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()

    def reviews(self, record, keys, decision, notes):
        current = self.client.get(f"/api/results/{record['id']}").get_json()
        version = current["version"]
        for key in keys:
            response = self.client.put(
                f"/api/results/{record['id']}/observations/{key}",
                json={"version": version, "decision": decision, "notes": notes},
            )
            self.assertEqual(response.status_code, 200)
            version = response.get_json()["version"]
        return version

    def test_flow(self):
        earlier = self.run_defects("diffuse")
        later = self.run_defects("parallel")
        keys = [item["key"] for item in earlier["metrics"]["observations"]][:4]
        self.reviews(earlier, keys, "confirm", "first pass")

        started = self.client.post(
            f"/api/results/{later['id']}/continuation",
            json={"version": later["version"], "earlier_id": earlier["id"],
                  "center_offset_px": 30, "footprint_overlap": 0.2},
        )
        self.assertEqual(started.status_code, 201)
        continuation = started.get_json()["continuation"]
        later_version = started.get_json()["version"]
        links = continuation["links"]
        earlier_keys = [item["earlier"] for item in links]
        later_keys = [item["later"] for item in links]

        self.assertEqual(len(earlier_keys), len(set(earlier_keys)), "each earlier spot continues at most once")
        self.assertEqual(len(later_keys), len(set(later_keys)), "each later spot continues at most once")
        summary = continuation["summary"]
        self.assertEqual(summary["continued"], len(links))
        self.assertEqual(
            summary["continued"] + summary["earlier_not_photographed"] + summary["earlier_no_qualifying_continuation"],
            summary["earlier_total"],
        )
        self.assertGreaterEqual(summary["confirmed_continued"], 1)
        topics = {row["outcome"] for row in continuation["archive"]["earlier"]}
        self.assertTrue(topics.issubset({"continued", "not_photographed", "no_qualifying_continuation"}))

        view = self.client.get(f"/api/results/{later['id']}/continuation").get_json()
        self.assertIsNotNone(view["earlier_measurement"])
        self.assertEqual(view["earlier_measurement"]["photo"]["role"], "diffuse")
        confirmed = {
            key: value for key, value in view["earlier_measurement"]["decisions"].items() if value["decision"] == "confirm"
        }
        self.assertEqual(len(confirmed), len(keys))
        self.assertTrue(all(value["notes"] == "first pass" for value in confirmed.values()))

        # The dialog can open the image of a carried earlier observation.
        carried = next(row for row in continuation["archive"]["earlier"] if row["outcome"] == "continued")
        image = self.client.get(
            f"/api/results/{earlier['id']}/observations/{carried['key']}/image"
        )
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.mimetype, "image/png")

        # Review the later run, then move one earlier spot; every review must survive.
        self.reviews(later, later_keys[:2], "confirm", "second pass")
        movable = next(
            (row for row in continuation["archive"]["earlier"] if row["outcome"] == "continued"), None
        )
        self.assertIsNotNone(movable)
        alternatives = [
            item for item in view["candidate_pairs"]
            if item["earlier"] == movable["key"] and item["later"] != movable["later_key"]
        ]
        if alternatives:
            target = alternatives[0]["later"]
        else:
            target = None
        previous = movable["later_key"]
        current_version = self.client.get(f"/api/results/{later['id']}").get_json()["version"]
        moved = self.client.put(
            f"/api/results/{later['id']}/continuation/links/{movable['key']}",
            json={"version": current_version, "later": target, "reason": "misread the footprint"},
        )
        self.assertEqual(moved.status_code, 200)
        updated = moved.get_json()["continuation"]
        moved_links = {item["earlier"]: item["later"] for item in updated["links"]}
        self.assertEqual(moved_links.get(movable["key"]), target)
        if target is not None:
            self.assertNotIn(target, [item for key, item in moved_links.items() if key != movable["key"]])
        self.assertEqual(len(moved_links), len(set(moved_links.values())))
        self.assertEqual(updated["history"][-1]["action"], "reassigned" if previous and target else "unlinked")
        self.assertEqual(updated["history"][-1]["reason"], "misread the footprint")

        earlier_record = self.client.get(f"/api/results/{earlier['id']}").get_json()
        self.assertEqual(len(earlier_record["decisions"]), len(keys))
        self.assertTrue(all(item["decision"] == "confirm" for item in earlier_record["decisions"].values()))
        later_record = self.client.get(f"/api/results/{later['id']}").get_json()
        self.assertEqual(
            [item["notes"] for item in later_record["decisions"].values()],
            ["second pass", "second pass"],
        )

        # A second later run may continue the same earlier run; a cycle is refused.
        third = self.run_defects("rake-left")
        another = self.client.post(
            f"/api/results/{third['id']}/continuation",
            json={"version": third["version"], "earlier_id": earlier["id"],
                  "center_offset_px": 30, "footprint_overlap": 0.2},
        )
        self.assertEqual(another.status_code, 201)
        cycle = self.client.post(
            f"/api/results/{earlier['id']}/continuation",
            json={"version": earlier_record["version"], "earlier_id": later["id"],
                  "center_offset_px": 30, "footprint_overlap": 0.2},
        )
        self.assertEqual(cycle.status_code, 422)

        # The continuation CSV lists every earlier spot and no removal.
        table = self.client.get(f"/api/results/{later['id']}/continuation.csv")
        self.assertEqual(table.status_code, 200)
        text = table.data.decode("utf-8-sig")
        self.assertIn("outcome", text.splitlines()[0])
        self.assertNotIn("removed", text.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
