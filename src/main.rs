use bevy::{prelude::*, window::{WindowMode, WindowResolution}};
use bevy_mod_picking::DefaultPickingPlugins;

mod board;
mod game_manager;
mod comun;
mod ia {
    pub mod monte_carlo_impl;
    pub mod brain;
}

use board::setup_board;
use crate::board::{create_pieces, HighlightedSquare, LegalMovesForPieceResource, PieceClicked, PieceIdByEntity, game_flow, button_system};
use crate::game_manager::GameManager;

#[derive(Resource, Default)]
struct GameManagerResource(GameManager);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window { 
                    title: "Dark Queen Zer0".to_string(),
                    resolution: WindowResolution::new(600., 650.),
                    mode: WindowMode::Windowed,
                    resizable: false,
                    ..default()
                }),
            ..default()
        }))
        .init_resource::<PieceClicked>()
        .init_resource::<PieceIdByEntity>()
        .init_resource::<LegalMovesForPieceResource>()
        .insert_resource(GameManagerResource(GameManager::new()))
        .init_resource::<HighlightedSquare>()
        .add_systems(Startup, (setup, setup_board, create_pieces))
        .add_systems(Update, (game_flow, button_system))
        .add_plugins(DefaultPickingPlugins)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}
