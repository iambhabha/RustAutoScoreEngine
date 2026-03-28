use std::collections::HashMap;

pub struct ScoringConfig {
    pub r_double: f32,
    pub r_treble: f32,
    pub r_outer_bull: f32,
    pub r_inner_bull: f32,
    pub w_double_treble: f32,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            r_double: 0.170,
            r_treble: 0.1074,
            r_outer_bull: 0.0159,
            r_inner_bull: 0.00635,
            w_double_treble: 0.01,
        }
    }
}

pub fn get_board_dict() -> HashMap<i32, &'static str> {
    let mut m = HashMap::new();
    // BDO standard mapping based on degrees
    let slices = ["13", "4", "18", "1", "20", "5", "12", "9", "14", "11", "8", "16", "7", "19", "3", "17", "2", "15", "10", "6"];
    for (i, &s) in slices.iter().enumerate() {
        m.insert(i as i32, s);
    }
    m
}

pub fn calculate_dart_score(cal_pts: &[[f32; 2]], dart_pt: &[f32; 2], config: &ScoringConfig) -> (i32, String) {
    // 1. Calculate Center (Average of 4 calibration points)
    let cx = cal_pts.iter().map(|p| p[0]).sum::<f32>() / 4.0;
    let cy = cal_pts.iter().map(|p| p[1]).sum::<f32>() / 4.0;

    // 2. Calculate average radius to boundary (doubles wire)
    let avg_r_px = cal_pts.iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
        .sum::<f32>() / 4.0;

    // 3. Relative distance of dart from center
    let dx = dart_pt[0] - cx;
    let dy = dart_pt[1] - cy;
    let dist_px = (dx.powi(2) + dy.powi(2)).sqrt();
    
    // Scale distance relative to BDO double radius
    let dist_scaled = (dist_px / avg_r_px) * config.r_double;

    // 4. Calculate Angle (0 is 3 o'clock, CCW)
    let mut angle_deg = (-dy).atan2(dx).to_degrees();
    if angle_deg < 0.0 { angle_deg += 360.0; }
    
    // Board is rotated such that 20 is at top (90 deg)
    // Sector width is 18 deg. Sector 20 is centered at 90 deg.
    // 90 deg is index 4 in slices (13, 4, 18, 1, 20...)
    // Each index is 18 deg. Offset = 4 * 18 = 72? No.
    // Let's use the standard mapping: (angle / 18)
    // Wait, the BOARD_DICT in Python uses int(angle / 18) where angle is 0-360.
    // We need to match the slice orientation.
    let board_dict = get_board_dict();
    let sector_idx = ((angle_deg / 18.0).floor() as i32) % 20;
    let sector_num = board_dict.get(&sector_idx).unwrap_or(&"0");

    // 5. Determine multipliers based on scaled distance
    let r_t = config.r_treble;
    let r_d = config.r_double;
    let w = config.w_double_treble;
    let r_ib = config.r_inner_bull;
    let r_ob = config.r_outer_bull;

    if dist_scaled > r_d {
        (0, "Miss".to_string())
    } else if dist_scaled <= r_ib {
        (50, "DB".to_string())
    } else if dist_scaled <= r_ob {
        (25, "B".to_string())
    } else if dist_scaled <= r_d && dist_scaled > (r_d - w) {
        let val = sector_num.parse::<i32>().unwrap_or(0);
        (val * 2, format!("D{}", sector_num))
    } else if dist_scaled <= r_t && dist_scaled > (r_t - w) {
        let val = sector_num.parse::<i32>().unwrap_or(0);
        (val * 3, format!("T{}", sector_num))
    } else {
        let val = sector_num.parse::<i32>().unwrap_or(0);
        (val, sector_num.to_string())
    }
}
