use bevy::{prelude::*, window::{WindowMode, WindowResolution}};
use bevy_mod_picking::DefaultPickingPlugins;

mod board;
mod game_manager;
mod comun;
mod rules;
mod train;
mod arena;
mod ia {
    pub mod monte_carlo_impl;
    pub mod brain;
    pub mod upper_confidence_bounds;
    pub mod encoding;
    pub mod neural;
    pub mod mcts;
}

use board::setup_board;
use crate::board::{create_pieces, HighlightedSquare, LegalMovesForPieceResource, PieceClicked, PieceIdByEntity, AiTask, game_flow, setup_ui, menu_system, update_menu_colors, status_system, start_computer_move, finish_computer_move};
use crate::game_manager::GameManager;

#[derive(Resource, Default)]
struct GameManagerResource(GameManager);

fn main() {
    // Headless subcommands: `cargo run --release -- train` / `arena`.
    match std::env::args().nth(1).as_deref() {
        Some("train") => {
            train::run();
            return;
        }
        Some("arena") => {
            arena::run();
            return;
        }
        _ => {}
    }

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window { 
                    title: "Dark Queen Zer0".to_string(),
                    resolution: WindowResolution::new(820., 650.),
                    mode: WindowMode::Windowed,
                    resizable: false,
                    ..default()
                }),
            ..default()
        }))
        .init_resource::<PieceClicked>()
        .init_resource::<PieceIdByEntity>()
        .init_resource::<LegalMovesForPieceResource>()
        .init_resource::<AiTask>()
        .insert_resource(GameManagerResource(GameManager::new()))
        .init_resource::<HighlightedSquare>()
        .add_systems(Startup, (setup, setup_board, create_pieces, setup_ui))
        .add_systems(Update, (game_flow, update_menu_colors, menu_system, status_system, finish_computer_move, start_computer_move).chain())
        .add_plugins(DefaultPickingPlugins)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}
