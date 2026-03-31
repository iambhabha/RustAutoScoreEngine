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
    let slices = [
        "6", "13", "4", "18", "1", "20", "5", "12", "9", "14", "11", "8", "16", "7", "19", "3",
        "17", "2", "15", "10",
    ];
    for (i, &s) in slices.iter().enumerate() {
        m.insert(i as i32, s);
    }
    m
}

pub fn calculate_dart_score(
    cal_pts: &[[f32; 2]],
    dart_pt: &[f32; 2],
    config: &ScoringConfig,
) -> (i32, String) {
    // 1. Calculate Center (Intersection of diagonals for better perspective handling)
    // Points: 0:TL, 1:TR, 2:BR, 3:BL
    let p1 = cal_pts[0]; // TL
    let p2 = cal_pts[1]; // TR
    let p3 = cal_pts[2]; // BR
    let p4 = cal_pts[3]; // BL

    // Line 1: p1 -> p3, Line 2: p2 -> p4
    // Using line intersection formula
    let den = (p1[0] - p3[0]) * (p2[1] - p4[1]) - (p1[1] - p3[1]) * (p2[0] - p4[0]);
    
    let (cx, cy) = if den.abs() > 1e-6 {
        let t = ((p1[0] - p2[0]) * (p2[1] - p4[1]) - (p1[1] - p2[1]) * (p2[0] - p4[0])) / den;
        (p1[0] + t * (p3[0] - p1[0]), p1[1] + t * (p3[1] - p1[1]))
    } else {
        // Fallback to average if parallel (shouldn't happen for a board)
        (cal_pts.iter().map(|p| p[0]).sum::<f32>() / 4.0, 
         cal_pts.iter().map(|p| p[1]).sum::<f32>() / 4.0)
    };

    // 2. Calculate average radius to boundary
    // We use the distance from center to each calibration point
    let avg_r_px = cal_pts
        .iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
        .sum::<f32>()
        / 4.0;

    // 3. Relative distance of dart from center
    let dx = dart_pt[0] - cx;
    let dy = dart_pt[1] - cy;
    let dist_px = (dx.powi(2) + dy.powi(2)).sqrt();

    // Scale distance relative to BDO double radius
    let dist_scaled = (dist_px / avg_r_px) * config.r_double;

    // 4. Calculate Angle (0 is 3 o'clock, CCW)
    let mut angle_deg = (-dy).atan2(dx).to_degrees();
    if angle_deg < 0.0 {
        angle_deg += 360.0;
    }

    // Center sectors by adding 9 degrees (half-sector width)
    let board_dict = get_board_dict();
    let sector_idx = (((angle_deg + 9.0) / 18.0).floor() as i32) % 20;
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
