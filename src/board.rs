use std::collections::HashMap;

use bevy::{
    ecs::{component::Component, system::Commands},
    math::{Vec2, Vec3},
    prelude::default,
    render::color::Color,
    sprite::{Sprite, SpriteBundle},
    transform::components::Transform,
    window::CursorIcon
};
use bevy::asset::{AssetServer, Handle};
use bevy::prelude::{AlignItems, BackgroundColor, BuildChildren, ButtonBundle, Changed, ChildBuilder, DespawnRecursiveExt, Entity, FlexDirection, Font, Image, Interaction, JustifyContent, NodeBundle, Query, Res, ResMut, Resource, Text, TextBundle, TextStyle, UiRect, With, Without};
use bevy::ui::{PositionType, Style, Val};
use bevy::window::Window;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;
use bevy_mod_picking::events::{Click, Pointer};
use bevy_mod_picking::PickableBundle;
use bevy_mod_picking::prelude::{Listener, On};

use crate::comun::{BandPlayer, BLACK_PAWN, BLACK_QUEEN, BOARD_SIZE, GamePlayer, GameStatus, Move, Square, Turn, WHITE_PAWN, WHITE_QUEEN};
use crate::game_manager::{BrainType, GameManager};
use crate::GameManagerResource;

const CELL_SIZE: f32 = 70.0;
const BOARD_X_OFFSET: f32 = -245.0;
const BOARD_Y_OFFSET: f32 = -265.0; // Offset to center the board on screen

const WHITE_PIECE_PATH: &str = "models/pieces/Roja.png";
const BLACK_PIECE_PATH: &str = "models/pieces/Negra.png";
const WHITE_QUEEN_PATH: &str = "models/pieces/Dama_roja.png";
const BLACK_QUEEN_PATH: &str = "models/pieces/Dama_Negra.png";

/// Whether the board should be drawn from Black's point of view. This happens
/// when the computer takes Red, so the human (Black) sees their own pieces at
/// the bottom of the screen.
fn board_is_flipped(game_manager: &GameManager) -> bool {
    game_manager.get_band_player().red == GamePlayer::Computer
}

/// World position of a board cell, honouring the current perspective. Row 0 is
/// at the bottom unless the board is flipped.
fn cell_translation(row: usize, col: usize, flipped: bool) -> Vec3 {
    let visual_row = if flipped { BOARD_SIZE - 1 - row } else { row };
    Vec3::new(
        col as f32 * CELL_SIZE + BOARD_X_OFFSET,
        visual_row as f32 * CELL_SIZE + BOARD_Y_OFFSET,
        0.0,
    )
}

#[derive(Resource, Default)]
pub struct HighlightedSquare {
    squares: Vec<Square>,
}

#[derive(Component, Debug, Default)]
pub struct Piece {
    pub piece_type: i8,
    pub id: usize
}

#[derive(Resource, Default)]
pub struct PieceClicked {
    clicked: bool,
    square: Square,
    piece: Piece
}

#[derive(Resource, Default)]
pub struct LegalMovesForPieceResource {
    pub moves: Vec<Move>
}

#[derive(Resource, Default)]
pub struct PieceIdByEntity(HashMap<usize, Entity>);

/// Background task that computes the AI move.
#[derive(Resource, Default)]
pub struct AiTask {
    task: Option<Task<Move>>,
}

#[derive(Component)]
pub struct WinnerWindow;

#[derive(Component)]
pub struct StatusText;

/// Colours for the menu buttons.
const MENU_BASE: Color = Color::rgb(0.30, 0.35, 0.50);
const MENU_HOVER: Color = Color::rgb(0.40, 0.45, 0.60);
const BRAIN_BASE: Color = Color::rgb(0.35, 0.30, 0.50);
const BRAIN_HOVER: Color = Color::rgb(0.45, 0.40, 0.60);
const WINNER_BASE: Color = Color::rgb(0.50, 0.50, 0.50);
const WINNER_HOVER: Color = Color::rgb(0.60, 0.60, 0.60);
const MODE_ACTIVE_BASE: Color = Color::rgb(0.20, 0.55, 0.30);
const MODE_ACTIVE_HOVER: Color = Color::rgb(0.30, 0.68, 0.40);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MenuAction {
    NewGame,
    /// New game against the computer, the human playing Red and moving first.
    HumanStarts,
    /// New game against the computer, the computer playing Red and moving
    /// first. The board is mirrored so the human (Black) is at the bottom.
    AiStarts,
    TwoPlayers,
    ToggleBrain,
}

/// The three selectable configurations, used to highlight the active one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PlayMode {
    HumanStarts,
    AiStarts,
    TwoPlayers,
}

/// Play mode selected by a button, or `None` for pure action buttons.
fn play_mode_of(action: MenuAction) -> Option<PlayMode> {
    match action {
        MenuAction::HumanStarts => Some(PlayMode::HumanStarts),
        MenuAction::AiStarts => Some(PlayMode::AiStarts),
        MenuAction::TwoPlayers => Some(PlayMode::TwoPlayers),
        MenuAction::NewGame | MenuAction::ToggleBrain => None,
    }
}

/// Play mode currently configured by the player band.
fn active_play_mode(band: &BandPlayer) -> Option<PlayMode> {
    match (band.red, band.black) {
        (GamePlayer::Human, GamePlayer::Computer) => Some(PlayMode::HumanStarts),
        (GamePlayer::Computer, GamePlayer::Human) => Some(PlayMode::AiStarts),
        (GamePlayer::Human, GamePlayer::Human) => Some(PlayMode::TwoPlayers),
        _ => None,
    }
}

#[derive(Component)]
pub struct MenuButton {
    pub action: MenuAction,
    /// Idle background colour, used unless this button is the active mode.
    pub base_color: Color,
    /// Background colour while hovered or pressed.
    pub hover_color: Color,
}

/// Label of the button that toggles the computer AI.
#[derive(Component)]
pub struct BrainButtonLabel;

// Set up the board
pub fn setup_board(mut commands: Commands, game_manager: Res<GameManagerResource>) {
    let flipped = board_is_flipped(&game_manager.0);
    for row in 0..BOARD_SIZE {
        for col in 0..BOARD_SIZE {
            commands.spawn((SpriteBundle {
                sprite: Sprite {
                    color: if (row + col) % 2 == 0 {
                        Color::OLIVE
                    } else {
                        Color::BEIGE
                    },
                    custom_size: Some(Vec2::new(CELL_SIZE, CELL_SIZE)),
                    ..default()
                },
                transform: Transform::from_translation(cell_translation(row, col, flipped)),
                ..Default::default()
            },
                            Square { row, col },
                            PickableBundle::default(),
                            On::<Pointer<Click>>::run(square_click_event)
            ));
        }
    }
}

pub fn create_pieces(mut commands: Commands, asset_server: Res<AssetServer>, game_manager: Res<GameManagerResource>, mut square_by_id: ResMut<PieceIdByEntity>) {
    let board = *game_manager.0.get_board();
    let flipped = board_is_flipped(&game_manager.0);
    spawn_pieces(&mut commands, &asset_server, &board, &mut square_by_id, flipped);
}

/// Spawns the piece entities for the given board.
pub fn spawn_pieces(
    commands: &mut Commands,
    asset_server: &AssetServer,
    board: &[[i8; BOARD_SIZE]; BOARD_SIZE],
    square_by_id: &mut PieceIdByEntity,
    flipped: bool,
) {
    let white_piece: Handle<Image> = asset_server.load(WHITE_PIECE_PATH);
    let black_piece: Handle<Image> = asset_server.load(BLACK_PIECE_PATH);
    let white_queen: Handle<Image> = asset_server.load(WHITE_QUEEN_PATH);
    let black_queen: Handle<Image> = asset_server.load(BLACK_QUEEN_PATH);

    for (row, row_values) in board.iter().enumerate() {
        for (col, &piece) in row_values.iter().enumerate() {
            let texture = match piece {
                WHITE_PAWN => Some(white_piece.clone()),
                BLACK_PAWN => Some(black_piece.clone()),
                WHITE_QUEEN => Some(white_queen.clone()),
                BLACK_QUEEN => Some(black_queen.clone()),
                _ => None,
            };

            if let Some(texture) = texture {
                let mut transform = Transform::from_translation(cell_translation(row, col, flipped));
                transform.translation.z = 0.1;
                transform.scale = Vec3::splat(0.4);

                let entity = commands
                    .spawn((
                        SpriteBundle {
                            texture,
                            transform,
                            ..Default::default()
                        },
                        Square { row, col },
                        Piece { piece_type: piece, id: (row * 10) + col },
                        PickableBundle::default(),
                        On::<Pointer<Click>>::run(piece_click_event),
                    ))
                    .id();

                square_by_id.0.insert((row * 10) + col, entity);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// UI: top bar with status and menu buttons
// ---------------------------------------------------------------------------

pub fn setup_ui(mut commands: Commands, asset_server: Res<AssetServer>, game_manager: Res<GameManagerResource>) {
    let font: Handle<Font> = asset_server.load("fonts/FiraSans-Bold.ttf");

    commands
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(72.0),
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::SpaceBetween,
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            },
            background_color: Color::rgb(0.12, 0.12, 0.14).into(),
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                TextBundle::from_section(
                    "Red's turn",
                    TextStyle {
                        font: font.clone(),
                        font_size: 20.0,
                        color: Color::WHITE,
                    },
                ),
                StatusText,
            ));

            parent
                .spawn(NodeBundle {
                    style: Style {
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    ..default()
                })
                .with_children(|row| {
                    spawn_menu_button(row, &font, "New game", MenuAction::NewGame);
                    spawn_menu_button(row, &font, "You start", MenuAction::HumanStarts);
                    spawn_menu_button(row, &font, "AI starts", MenuAction::AiStarts);
                    spawn_menu_button(row, &font, "2 Players", MenuAction::TwoPlayers);
                    spawn_brain_button(row, &font, brain_label(game_manager.0.get_brain_type()));
                });
        });
}

fn spawn_menu_button(parent: &mut ChildBuilder, font: &Handle<Font>, label: &str, action: MenuAction) {
    parent
        .spawn((
            ButtonBundle {
                style: Style {
                    width: Val::Px(105.0),
                    height: Val::Px(44.0),
                    margin: UiRect::all(Val::Px(3.0)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                background_color: MENU_BASE.into(),
                ..default()
            },
            MenuButton { action, base_color: MENU_BASE, hover_color: MENU_HOVER },
        ))
        .with_children(|parent| {
            parent.spawn(TextBundle::from_section(
                label,
                TextStyle {
                    font: font.clone(),
                    font_size: 15.0,
                    color: Color::WHITE,
                },
            ));
        });
}

/// Text of the button that toggles the computer AI.
fn brain_label(brain: BrainType) -> &'static str {
    match brain {
        BrainType::Neural => "AI: Neural",
        _ => "AI: Classic",
    }
}

fn spawn_brain_button(parent: &mut ChildBuilder, font: &Handle<Font>, label: &str) {
    parent
        .spawn((
            ButtonBundle {
                style: Style {
                    width: Val::Px(125.0),
                    height: Val::Px(44.0),
                    margin: UiRect::all(Val::Px(3.0)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                background_color: BRAIN_BASE.into(),
                ..default()
            },
            MenuButton { action: MenuAction::ToggleBrain, base_color: BRAIN_BASE, hover_color: BRAIN_HOVER },
        ))
        .with_children(|parent| {
            parent.spawn((
                TextBundle::from_section(
                    label,
                    TextStyle {
                        font: font.clone(),
                        font_size: 15.0,
                        color: Color::WHITE,
                    },
                ),
                BrainButtonLabel,
            ));
        });
}

/// Updates the status text with the turn or the result.
pub fn status_system(game_manager: Res<GameManagerResource>, mut text_query: Query<&mut Text, With<StatusText>>) {
    let band = game_manager.0.get_band_player();
    let message = match game_manager.0.get_game_status() {
        GameStatus::Playing => {
            let (name, player) = match game_manager.0.get_turn() {
                Turn::Red => ("Red", band.red),
                Turn::Black => ("Black", band.black),
            };
            match player {
                GamePlayer::Computer => "AI is thinking...".to_string(),
                GamePlayer::Human => format!("{name}'s turn"),
            }
        }
        GameStatus::RedWins => "Red wins!".to_string(),
        GameStatus::BlackWins => "Black wins!".to_string(),
        GameStatus::Draw => "Draw".to_string(),
    };

    if let Ok(mut text) = text_query.get_single_mut() {
        if text.sections[0].value != message {
            text.sections[0].value = message;
        }
    }
}

/// Handles the menu buttons: switches mode or restarts the game.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn menu_system(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut game_manager: ResMut<GameManagerResource>,
    interactions: Query<(&Interaction, &MenuButton), (Changed<Interaction>, With<MenuButton>)>,
    pieces: Query<Entity, With<Piece>>,
    winner_windows: Query<Entity, With<WinnerWindow>>,
    mut piece_by_id: ResMut<PieceIdByEntity>,
    mut squares: Query<(&mut Sprite, &mut Transform, &Square), Without<Piece>>,
    mut highlighted: ResMut<HighlightedSquare>,
    mut legal_moves: ResMut<LegalMovesForPieceResource>,
    mut ai_task: ResMut<AiTask>,
    mut brain_labels: Query<&mut Text, With<BrainButtonLabel>>,
) {
    for (interaction, button) in &interactions {
        if *interaction != Interaction::Pressed {
            continue;
        }

        match button.action {
            MenuAction::HumanStarts => game_manager.0.set_band_player(BandPlayer {
                red: GamePlayer::Human,
                black: GamePlayer::Computer,
            }),
            // The computer takes Red (which moves first) so selecting this mode
            // makes it play immediately; the board is then flipped so the human
            // (Black) is at the bottom.
            MenuAction::AiStarts => game_manager.0.set_band_player(BandPlayer {
                red: GamePlayer::Computer,
                black: GamePlayer::Human,
            }),
            MenuAction::TwoPlayers => game_manager.0.set_band_player(BandPlayer {
                red: GamePlayer::Human,
                black: GamePlayer::Human,
            }),
            MenuAction::ToggleBrain => {
                let next = match game_manager.0.get_brain_type() {
                    BrainType::Neural => BrainType::UpperConfidenceBounds,
                    _ => BrainType::Neural,
                };
                game_manager.0.set_brain_type(next);
            }
            MenuAction::NewGame => {}
        }

        if let Ok(mut text) = brain_labels.get_single_mut() {
            text.sections[0].value =
                brain_label(game_manager.0.get_brain_type()).to_string();
        }

        // Reset the logical state and rebuild the scene.
        ai_task.task = None;
        game_manager.0.reset();

        for entity in &pieces {
            commands.entity(entity).despawn_recursive();
        }
        for entity in &winner_windows {
            commands.entity(entity).despawn_recursive();
        }
        piece_by_id.0.clear();

        let flipped = board_is_flipped(&game_manager.0);
        for (mut sprite, mut transform, square) in &mut squares {
            *transform = Transform::from_translation(cell_translation(square.row, square.col, flipped));
            sprite.color = if (square.row + square.col) % 2 == 0 {
                Color::OLIVE
            } else {
                Color::BEIGE
            };
        }

        highlighted.squares.clear();
        legal_moves.moves.clear();

        let board = *game_manager.0.get_board();
        spawn_pieces(&mut commands, &asset_server, &board, &mut piece_by_id, flipped);
    }
}

/// Keeps the mode buttons coloured so the active configuration is visible, and
/// applies the hover colour to every menu button.
pub fn update_menu_colors(
    game_manager: Res<GameManagerResource>,
    mut buttons: Query<(&MenuButton, &Interaction, &mut BackgroundColor)>,
) {
    let active = active_play_mode(game_manager.0.get_band_player());

    for (button, interaction, mut color) in &mut buttons {
        let is_active = play_mode_of(button.action) == active && active.is_some();
        let (base, hover) = if is_active {
            (MODE_ACTIVE_BASE, MODE_ACTIVE_HOVER)
        } else {
            (button.base_color, button.hover_color)
        };

        let desired = match *interaction {
            Interaction::None => base,
            Interaction::Hovered | Interaction::Pressed => hover,
        };

        if color.0 != desired {
            color.0 = desired;
        }
    }
}



#[allow(clippy::too_many_arguments)]
fn square_click_event(
    event: Listener<Pointer<Click>>,
    asset_server: Res<AssetServer>,
    mut transform_square_piece: Query<(&mut Handle<Image>, &mut Transform, &mut Square, &Piece)>,
    empty_square: Query<&Square, Without<Piece>>,
    mut squares_sprites: Query<(&mut Sprite, &Square), Without<Piece>>,
    mut highlighted_square: ResMut<HighlightedSquare>,
    mut piece_clicked: ResMut<PieceClicked>,
    mut game_manager: ResMut<GameManagerResource>,
    legal_moves_for_piece: ResMut<LegalMovesForPieceResource>,
    mut piece_by_id: ResMut<PieceIdByEntity>,
    mut commands: Commands,
) {
    let square_clicked = empty_square.get(event.target).unwrap();
    //println!("Square clicked: {:?}", square_clicked);

    if highlighted_square.squares.iter().any(|square| square.row == square_clicked.row && square.col == square_clicked.col)
        && piece_clicked.clicked {
            //println!("Piece clicked: {:?}", piece_clicked.square);
            piece_clicked.clicked = false;

            unhighlight_square(&mut squares_sprites, &mut highlighted_square.squares);

            if let Some(move_to) = legal_moves_for_piece.moves.iter().find(|&m| m.to.row == square_clicked.row && m.to.col == square_clicked.col) {
                move_piece(
                    move_to,
                    &mut transform_square_piece,
                    &asset_server,
                    &mut game_manager,
                    &mut piece_by_id,
                    &mut commands,
                );
            }
        }
}

fn piece_click_event(event: Listener<Pointer<Click>>,
                     pieces: Query<(&Square, &Piece)>,
                     mut piece_clicked: ResMut<PieceClicked>,
                     mut squares: Query<(&mut Sprite, &Square), Without<Piece>>,
                     mut game_manager: ResMut<GameManagerResource>,
                     mut highlighted_square: ResMut<HighlightedSquare>,
                     mut legal_moves_for_piece: ResMut<LegalMovesForPieceResource>
) {
    let (square_clicked, piece) = pieces.get(event.target).unwrap();

    //println!("Piece Clicked: {:?} - {:?}", square_clicked, piece);
    piece_clicked.clicked = true;
    piece_clicked.square.col = square_clicked.col;
    piece_clicked.square.row = square_clicked.row;
    piece_clicked.piece.piece_type = piece.piece_type;

    unhighlight_square(&mut squares, &mut highlighted_square.squares);
    legal_moves_for_piece.moves.clear();

    let legal_moves = game_manager.0.get_legal_moves();
    for legal_move in legal_moves {
        //println!("Legal move: {:?}", legal_move);
        if legal_move.from.row == square_clicked.row && legal_move.from.col == square_clicked.col {
            highlight_square(
                &legal_move.to,
                &mut squares,
                &mut highlighted_square.squares
            );
            legal_moves_for_piece.moves.push(*legal_move);
        }
    }
}

fn move_piece(
    move_to: &Move,
    transform_square_piece: &mut Query<(&mut Handle<Image>, &mut Transform, &mut Square, &Piece)>,
    asset_server: &Res<AssetServer>,
    game_manager: &mut ResMut<GameManagerResource>,
    piece_by_id: &mut ResMut<PieceIdByEntity>,
    commands: &mut Commands,
) {
    let mut found_piece = None;
    for (texture, transform, square_piece, _) in transform_square_piece.iter_mut() {
        if square_piece.row == move_to.from.row && square_piece.col == move_to.from.col {
            found_piece = Some((texture, transform, square_piece));
            break;
        }
    }

    if let Some((texture, mut transform, mut square_piece)) = found_piece {
        let flipped = board_is_flipped(&game_manager.0);
        let target = cell_translation(move_to.to.row, move_to.to.col, flipped);
        transform.translation.x = target.x;
        transform.translation.y = target.y;

        game_manager.0.do_move(*move_to);

        square_piece.row = move_to.to.row;
        square_piece.col = move_to.to.col;

        if move_to.promotion {
            let queen_texture: Handle<Image> = if move_to.turn == Turn::Red {
                asset_server.load(WHITE_QUEEN_PATH).clone()
            } else {
                asset_server.load(BLACK_QUEEN_PATH).clone()
            };

            let texture = texture.into_inner();
            *texture = queen_texture;
        }

        if let Some(square_eaten) = move_to.eat {
            let id_opt = get_id_from_square(square_eaten.0, transform_square_piece);
            if let Some(id) = id_opt {
                if let Some(entity) = piece_by_id.0.get(&id) {
                    commands.entity(*entity).despawn();
                    piece_by_id.0.remove(&id);
                }
            }
        }
    }
}

fn highlight_square(dest_square: &Square, sprite_square: &mut Query<(&mut Sprite, &Square), Without<Piece>>, highlighted_square: &mut Vec<Square>) {
    for (mut sprite, square) in sprite_square.iter_mut() {
        if square.row == dest_square.row && square.col == dest_square.col {
            sprite.color = Color::GREEN;
            highlighted_square.push(*dest_square);
            break;
        }
    }
}

fn unhighlight_square(squares: &mut Query<(&mut Sprite, &Square), Without<Piece>>,  highlighted_square: &mut Vec<Square>) {
    for highlitghted_square in highlighted_square.iter() {
        for (mut sprite, square) in squares.iter_mut() {
            if square.row == highlitghted_square.row && square.col == highlitghted_square.col {
                sprite.color = Color::OLIVE;
            }
        }
    }
    highlighted_square.clear();
}

fn get_id_from_square(square: Square, pieces: &mut Query<(&mut Handle<Image>, &mut Transform, &mut Square, &Piece)>) -> Option<usize> {
    for (_, _, s, piece) in pieces.iter() {
        if s.row == square.row && s.col == square.col {
            return Some(piece.id);
        }
    }

    None
}

pub fn game_flow(
    mut commands: Commands,
    game_manager: Res<GameManagerResource>,
    asset_server: Res<AssetServer>,
    query: Query<Entity, With<WinnerWindow>>,
) {
    if game_manager.0.get_game_status() != GameStatus::Playing {
        let winner_name = match game_manager.0.get_game_status() {
            GameStatus::RedWins => "Red wins!!",
            GameStatus::BlackWins => "Black wins!!",
            _ => "Draw!!",
        };

        if query.is_empty() {
            commands
                .spawn((
                    NodeBundle {
                        style: Style {
                            width: Val::Px(300.0),
                            height: Val::Px(200.0),
                            position_type: PositionType::Absolute,
                            left: Val::Percent(50.0),
                            top: Val::Percent(50.0),
                            //transform: Transform::from_xyz(-150.0, -100.0, 0.0),
                            flex_direction: FlexDirection::Column,
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        background_color: Color::rgb(0.8, 0.8, 0.8).into(),
                        ..default()
                    },
                    WinnerWindow,
                ))
                .with_children(|parent| {
                    parent.spawn(TextBundle::from_section(
                        winner_name.to_string(),
                        TextStyle {
                            font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                            font_size: 24.0,
                            color: Color::BLACK,
                        },
                    ));
                    parent.spawn((
                        ButtonBundle {
                            style: Style {
                                width: Val::Px(100.0),
                                height: Val::Px(50.0),
                                margin: UiRect::all(Val::Px(10.0)),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                ..default()
                            },
                            background_color: Color::rgb(0.5, 0.5, 0.5).into(),
                            ..default()
                        },
                        MenuButton { action: MenuAction::NewGame, base_color: WINNER_BASE, hover_color: WINNER_HOVER },
                    ))
                        .with_children(|parent| {
                            parent.spawn(TextBundle::from_section(
                                "New game",
                                TextStyle {
                                    font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                                    font_size: 20.0,
                                    color: Color::WHITE,
                                },
                            ));
                        });
                });
        }
    }
    else {
        for entity in &query {
            commands.entity(entity).despawn_recursive();
        }
    }
}

/// Starts the background AI move computation if it is the computer's turn and
/// no task is already running.
pub fn start_computer_move(
    game_manager: Res<GameManagerResource>,
    mut ai_task: ResMut<AiTask>,
    mut window: Query<&mut Window>,
) {
    if ai_task.task.is_some() || game_manager.0.get_game_status() != GameStatus::Playing {
        return;
    }

    let current_player = match game_manager.0.get_turn() {
        Turn::Red => game_manager.0.get_band_player().red,
        Turn::Black => game_manager.0.get_band_player().black,
    };
    if current_player != GamePlayer::Computer {
        return;
    }

    let mut snapshot = game_manager.0.clone();
    let task = AsyncComputeTaskPool::get().spawn(async move { snapshot.computer_plays() });
    ai_task.task = Some(task);

    if let Ok(mut window) = window.get_single_mut() {
        window.cursor.icon = CursorIcon::Wait;
    }
}

/// Collects the AI-computed move and applies it to the board.
pub fn finish_computer_move(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut ai_task: ResMut<AiTask>,
    mut game_manager: ResMut<GameManagerResource>,
    mut transform_square_piece: Query<(&mut Handle<Image>, &mut Transform, &mut Square, &Piece)>,
    mut piece_by_id: ResMut<PieceIdByEntity>,
    mut window: Query<&mut Window>,
) {
    let Some(task) = ai_task.task.as_mut() else {
        return;
    };

    if let Some(movement) = future::block_on(future::poll_once(task)) {
        ai_task.task = None;
        move_piece(
            &movement,
            &mut transform_square_piece,
            &asset_server,
            &mut game_manager,
            &mut piece_by_id,
            &mut commands,
        );

        if let Ok(mut window) = window.get_single_mut() {
            window.cursor.icon = CursorIcon::Default;
        }
    }
}
