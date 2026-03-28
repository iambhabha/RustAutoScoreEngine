pub enum Command {
    Gui,
    Test { img_path: String },
    Train,
}

pub struct AppArgs {
    pub command: Command,
}

impl AppArgs {
    pub fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();

        if args.len() > 1 {
            match args[1].as_str() {
                "gui" => Self {
                    command: Command::Gui,
                },
                "test" => {
                    let img_path = if args.len() > 2 {
                        args[2].clone()
                    } else {
                        "test.jpg".to_string()
                    };
                    Self {
                        command: Command::Test { img_path },
                    }
                }
                _ => Self {
                    command: Command::Train,
                },
            }
        } else {
            Self {
                command: Command::Train,
            }
        }
    }
}
