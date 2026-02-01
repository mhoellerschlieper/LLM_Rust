// train.rs
// Description: Dataset loading utilities (JSON and CSV) for training and fine tuning.
// History:
// - 2026-02-01: Consolidate dataset loader into train.rs.
// Author: Marcus Schlieper

use std::fs;

use csv::ReaderBuilder;

pub struct Dataset {
    pub pretraining_data: Vec<String>,
    pub chat_training_data: Vec<String>,
}

#[allow(dead_code)]
#[allow(clippy::upper_case_acronyms)]
pub enum DatasetType {
    JSON,
    CSV,
}

impl Dataset {
    pub fn new(s_pre_path: &str, s_chat_path: &str, e_type: DatasetType) -> Self {
        let pretraining_data: Vec<String>;
        let chat_training_data: Vec<String>;

        match e_type {
            DatasetType::CSV => {
                pretraining_data = get_data_from_csv(s_pre_path);
                chat_training_data = get_data_from_csv(s_chat_path);
            }
            DatasetType::JSON => {
                pretraining_data = get_data_from_json(s_pre_path);
                chat_training_data = get_data_from_json(s_chat_path);
            }
        }

        Self {
            pretraining_data,
            chat_training_data,
        }
    }
}

fn get_data_from_json(s_path: &str) -> Vec<String> {
    let s_json = fs::read_to_string(s_path).expect("Failed to read data file");
    let v_data: Vec<String> = serde_json::from_str(&s_json).expect("Failed to parse data file");
    v_data
}

fn get_data_from_csv(s_path: &str) -> Vec<String> {
    let file = fs::File::open(s_path).expect("Failed to open CSV file");
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

    let mut v_data: Vec<String> = Vec::new();
    for result in rdr.records() {
        let record = result.expect("Failed to read CSV record");
        v_data.push(record.iter().collect::<Vec<&str>>().join(","));
    }
    v_data
}