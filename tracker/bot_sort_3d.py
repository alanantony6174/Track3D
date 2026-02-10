from .bot_sort import BOTSORT, BOTrack, ReID
from .utils.kalman_filter_3d import KalmanFilter3D
from .utils.matching_3d import iou_3d, linear_assignment_3d, euclidean_distance_3d
from .utils import matching
import numpy as np
from .basetrack import TrackState

class BOTrack3D(BOTrack):
    shared_kalman = KalmanFilter3D(np.zeros(7)) # Dummy init

    def __init__(self, bbox3d, score, cls, feat=None, feat_history=50):
        # bbox3d: [x, y, z, theta, l, w, h]
        # Call parent with dummy 2D box; we override all KF logic anyway
        super().__init__(np.array([0, 0, 0, 0, 0]), score, cls, feat, feat_history)
        
        self.bbox3d = bbox3d
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.covariance = None
        self.is_activated = False
        self.history = [] # Store history of (x, y, z)
        self.velocity = np.zeros(3)

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        # Initialize 3D Kalman Filter
        # State: 10D: x, y, z, theta, l, w, h, vx, vy, vz
        self.kf = KalmanFilter3D(self.bbox3d, self.track_id)
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        
    def predict(self):
        # Update 3D KF
        self.kf.predict()
        
        # Sync state back to bbox3d for matching
        state = self.kf.get_state()
        self.bbox3d = state[:7] # Update current position estimate
        
    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        # Update appearance features (ReID)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        
        new_bbox3d = new_track.bbox3d
        
        # Update KF
        self.kf.update(new_bbox3d[:7])
        
        # Sync state
        state = self.kf.get_state()
        self.bbox3d = state[:7]
        self.velocity = self.kf.get_velocity()
        
        self.history.append(self.bbox3d[:3])
        if len(self.history) > 50:
            self.history.pop(0)
        
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        
    def re_activate(self, new_track, frame_id, new_id=False):
        self.frame_id = frame_id
        self.tracklet_len = 0
        
        # Update appearance features (ReID)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        
        new_bbox3d = new_track.bbox3d
        
        # Update KF
        self.kf.update(new_bbox3d[:7])
        
        # Sync state
        state = self.kf.get_state()
        self.bbox3d = state[:7]
        self.velocity = self.kf.get_velocity()
        
        self.history.append(self.bbox3d[:3])
        if len(self.history) > 50:
            self.history.pop(0)
        
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        
        if new_id:
            self.track_id = self.next_id()

    @property
    def result(self):
        # Return 3D result: [h, w, l, x, y, z, theta, ID, vx, vy, vz]
        state = self.kf.get_state()
        velocity = self.kf.get_velocity()
        
        # Returning: x, y, z, l, w, h, theta, id, score, class, vx, vy, vz
        res = [*state[:7], self.track_id, self.score, self.cls]
        res.extend(velocity)
        return res
        
    def get_history(self):
        return self.history

class BOTSORT_3D(BOTSORT):
    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)
        
        # ReID encoder (reuse parent's encoder if with_reid is set)
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.second_match_thresh = getattr(args, 'second_match_thresh', 0.25)
        self.unconfirmed_thresh = getattr(args, 'unconfirmed_thresh', 0.35)
        
    def init_track(self, results, img=None):
        """Initialize BOTrack3D objects from 3D detection dicts."""
        # If ReID is enabled and we have an image, extract features from 2D crops
        feats = None
        if self.args.with_reid and self.encoder is not None and img is not None:
            # Build 2D bboxes for crop extraction: [cx, cy, w, h, idx]
            bboxes_2d = []
            for i, res in enumerate(results):
                if 'bbox_2d' in res and res['bbox_2d'] is not None:
                    cx, cy, bw, bh = res['bbox_2d']
                    bboxes_2d.append([cx, cy, bw, bh, i])
            if len(bboxes_2d) > 0:
                bboxes_2d = np.array(bboxes_2d)
                feats = self.encoder(img, bboxes_2d)
        
        tracks = []
        for i, res in enumerate(results):
            feat = feats[i] if feats is not None and i < len(feats) else res.get('feat')
            track = BOTrack3D(res['bbox3d'], res['score'], res['cls'], feat)
            tracks.append(track)
        return tracks

    def get_dists(self, tracks, detections):
        """3D distance cost matrix, fused with ReID embedding distance when available."""
        tl = [t.bbox3d for t in tracks]
        dl = [d.bbox3d for d in detections]
        
        if len(tl) == 0 or len(dl) == 0:
            return np.zeros((len(tl), len(dl)))
            
        dists = euclidean_distance_3d(np.array(tl), np.array(dl))
        
        # Normalize 3D distance to [0, 1] range for fusion with appearance
        # Distances > 2m are almost certainly different people
        dists_norm = np.minimum(dists / 2.0, 1.0)
        
        # Gate: tracks too far apart can't be the same person
        dists_mask = dists_norm > (1 - self.proximity_thresh)
        
        # Fuse with ReID embeddings if available
        if self.args.with_reid and self.encoder is not None:
            # Check if tracks and detections have features
            has_feats = (all(t.smooth_feat is not None for t in tracks) and
                        all(d.curr_feat is not None for d in detections))
            if has_feats:
                emb_dists = matching.embedding_distance(tracks, detections) / 2.0
                emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
                emb_dists[dists_mask] = 1.0  # Don't use appearance for distant tracks
                dists_norm = np.minimum(dists_norm, emb_dists)
        
        return dists_norm
    
    def update(self, results, img=None):
        """Full BoT-SORT update with two-stage association and unconfirmed track handling."""
        self.frame_id += 1
        
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # ============================================================
        # Step 1: Split detections by confidence
        # ============================================================
        results_first = [r for r in results if r['score'] >= self.args.track_high_thresh]
        results_second = [r for r in results if self.args.track_low_thresh < r['score'] < self.args.track_high_thresh]
        
        detections = self.init_track(results_first, img)
        detections_second = self.init_track(results_second, img)
        
        # ============================================================
        # Step 2: Separate confirmed vs unconfirmed tracks
        # ============================================================
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # Build association pool: confirmed tracked + lost
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict all tracks
        for strack in strack_pool:
            strack.predict()
        for strack in unconfirmed:
            strack.predict()
            
        # ============================================================
        # Step 3: First association — high-confidence detections
        # ============================================================
        dists = self.get_dists(strack_pool, detections)
        
        # DEBUG
        if len(detections) > 0:
            print(f"Frame {self.frame_id}: {len(detections)} Hi-Conf + {len(detections_second)} Lo-Conf Dets, "
                  f"{len(strack_pool)} Pool, {len(unconfirmed)} Unconfirmed")
            for i, det in enumerate(detections):
                print(f"  Det {i}: [{det.bbox3d[0]:.3f}, {det.bbox3d[1]:.3f}, {det.bbox3d[2]:.3f}] Score: {det.score:.3f}")
            for trk in strack_pool[:5]:  # Print first 5 only
                print(f"  Trk {trk.track_id}: [{trk.bbox3d[0]:.3f}, {trk.bbox3d[1]:.3f}, {trk.bbox3d[2]:.3f}] Score: {trk.score:.3f}")
            if len(strack_pool) > 5:
                print(f"  ... and {len(strack_pool) - 5} more tracks")

        matches, u_track, u_detection = linear_assignment_3d(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        # ============================================================
        # Step 4: Second association — low-confidence detections
        #         Only match against remaining TRACKED (not lost) tracks
        # ============================================================
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        if len(r_tracked_stracks) > 0 and len(detections_second) > 0:
            dists_second = self.get_dists(r_tracked_stracks, detections_second)
            matches_second, u_track_second, _ = linear_assignment_3d(dists_second, thresh=self.second_match_thresh)
            
            for itracked, idet in matches_second:
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
                    
            # Mark remaining unmatched tracked stracks as lost
            for it in u_track_second:
                track = r_tracked_stracks[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            # No second-stage detections, mark all unmatched tracked as lost
            for it in u_track:
                track = strack_pool[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
                
        # ============================================================
        # Step 5: Third association — unconfirmed tracks
        #         Match remaining first-stage detections against unconfirmed
        # ============================================================
        detections_remain = [detections[i] for i in u_detection]
        
        if len(unconfirmed) > 0 and len(detections_remain) > 0:
            dists_unconfirmed = self.get_dists(unconfirmed, detections_remain)
            matches_unc, u_unconfirmed, u_detection_unc = linear_assignment_3d(dists_unconfirmed, thresh=self.unconfirmed_thresh)
            
            for itracked, idet in matches_unc:
                unconfirmed[itracked].update(detections_remain[idet], self.frame_id)
                activated_stracks.append(unconfirmed[itracked])
                
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)
                
            # Update remaining unmatched detections for new track init
            detections_remain = [detections_remain[i] for i in u_detection_unc]
        else:
            # Remove all unconfirmed that didn't match
            for track in unconfirmed:
                track.mark_removed()
                removed_stracks.append(track)
        
        # ============================================================
        # Step 6: Initialize new tracks from remaining detections
        # ============================================================
        for track in detections_remain:
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(None, self.frame_id)
            activated_stracks.append(track)

        # ============================================================
        # Step 7: Update state — remove expired lost tracks
        # ============================================================
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ============================================================
        # Step 8: Finalize track lists
        # ============================================================
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        
        # Proper duplicate removal (tracked vs lost)
        self.tracked_stracks, self.lost_stracks = self._remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
        
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]
        
        return [t for t in self.tracked_stracks if t.is_activated]

    @staticmethod
    def _remove_duplicate_stracks(stracksa, stracksb):
        """Remove duplicate stracks between tracked and lost lists using 3D distance."""
        if len(stracksa) == 0 or len(stracksb) == 0:
            return stracksa, stracksb
            
        # Compute pairwise 3D distances
        bboxes_a = np.array([t.bbox3d for t in stracksa])
        bboxes_b = np.array([t.bbox3d for t in stracksb])
        dists = euclidean_distance_3d(bboxes_a, bboxes_b)
        
        pairs = np.where(dists < 0.3)  # Within 30cm
        
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)  # Remove from lost (shorter lived)
            else:
                dupa.append(p)  # Remove from tracked (shorter lived)
                
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
