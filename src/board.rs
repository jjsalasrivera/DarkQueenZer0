use std::collections::HashMap;
use bevy::{
    ecs::{component::Component, system::Commands},
    math::{Vec2, Vec3},
    prelude::default,
    render::color::Color,
    sprite::{Sprite, SpriteBundle},
    transform::components::Transform,
};
use bevy::app::AppExit;
use bevy::asset::{AssetServer, Handle};
use bevy::prelude::{AlignItems, BackgroundColor, BuildChildren, ButtonBundle, Changed, DespawnRecursiveExt, Entity, EventWriter, FlexDirection, Image, Interaction, JustifyContent, NodeBundle, Query, Res, ResMut, Resource, TextBundle, TextStyle, UiRect, With, Without};
use bevy::ui::{PositionType, Style, Val};
use bevy_mod_picking::events::{Click, Pointer};
use bevy_mod_picking::PickableBundle;
use bevy_mod_picking::prelude::{Listener, On};
use crate::comun::{BLACK_PAWN, BOARD_SIZE, GamePlayer, GameStatus, Move, Square, Turn, WHITE_PAWN};
use crate::GameManagerResource;

const CELL_SIZE: f32 = 70.0;
const BOARD_X_OFFSET: f32 = -245.0;
const BOARD_Y_OFFSET: f32 = -265.0; // Desplazamiento para centrar el tablero en la pantalla

const WHITE_PIECE_PATH: &str = "models/pieces/Roja.png";
const BLACK_PIECE_PATH: &str = "models/pieces/Negra.png";
const WHITE_QUEEN_PATH: &str = "models/pieces/Dama_Roja.png";
const BLACK_QUEEN_PATH: &str = "models/pieces/Dama_Negra.png";

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

#[derive(Component)]
pub struct WinnerWindow;
#[derive(Component)]
pub struct AcceptButton;

// Función para inicializar el tablero
pub fn setup_board(mut commands: Commands) {
    for row in 0..BOARD_SIZE {
        for col in 0..BOARD_SIZE {
            let x = col as f32 * CELL_SIZE + BOARD_X_OFFSET;
            let y = row as f32 * CELL_SIZE + BOARD_Y_OFFSET;

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
                transform: Transform::from_xyz(x, y, 0.),
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
    let white_piece: Handle<Image> = asset_server.load(WHITE_PIECE_PATH);
    let black_piece: Handle<Image> = asset_server.load(BLACK_PIECE_PATH);

    let board = game_manager.0.get_board();

    for row in 0..BOARD_SIZE {
        for col in 0..BOARD_SIZE {
            let x = col as f32 * CELL_SIZE + BOARD_X_OFFSET;
            let y = row as f32 * CELL_SIZE + BOARD_Y_OFFSET;

            println!("Board[{}][{}]: {}", row, col, board[row][col]);

            let piece = board[row][col];
            let tupla = match piece {
                WHITE_PAWN => Some((SpriteBundle {
                    texture: white_piece.clone(),
                    transform: {
                        let mut transform = Transform::from_xyz(x, y, 0.1);
                        transform.scale = Vec3::splat(0.4);
                        transform
                    },
                    ..Default::default()
                },
                                              Square { row, col },
                                              Piece{ piece_type: WHITE_PAWN, id: (row * 10) + col },
                                              PickableBundle::default(),
                                              On::<Pointer<Click>>::run(piece_click_event)
                )),
                BLACK_PAWN => Some((SpriteBundle {
                    texture: black_piece.clone(),
                    transform: {
                        let mut transform = Transform::from_xyz(x, y, 0.1);
                        transform.scale = Vec3::splat(0.4);
                        transform
                    },
                    ..Default::default()
                },
                                              Square { row, col }, Piece{ piece_type: BLACK_PAWN, id: (row * 10) + col },
                                              PickableBundle::default(), On::<Pointer<Click>>::run(piece_click_event)
                )),
                _ => None
            };

            if let Some(tupla) = tupla {
                let id = commands.spawn(tupla).id();
                square_by_id.0.insert((row * 10 ) + col, id);
            }
        }
    }
}


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

    if highlighted_square.squares.iter().any(|square| square.row == square_clicked.row && square.col == square_clicked.col) {
        if piece_clicked.clicked {
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
            legal_moves_for_piece.moves.push(legal_move.clone());
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
        let delta_x = move_to.to.col as f32 * CELL_SIZE + BOARD_X_OFFSET - transform.translation.x;
        let delta_y = move_to.to.row as f32 * CELL_SIZE + BOARD_Y_OFFSET - transform.translation.y;

        transform.translation = transform.translation + Vec3::new(delta_x, delta_y, 0.1);
        game_manager.0.do_move(move_to.clone());

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
            highlighted_square.push(dest_square.clone());
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
    mut game_manager: ResMut<GameManagerResource>,
    asset_server: Res<AssetServer>,
    query: Query<Entity, With<WinnerWindow>>,
    mut transform_square_piece: Query<(&mut Handle<Image>, &mut Transform, &mut Square, &Piece)>,
    mut piece_by_id: ResMut<PieceIdByEntity>
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
                        format!("¡{}", winner_name),
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
                        AcceptButton,
                    ))
                        .with_children(|parent| {
                            parent.spawn(TextBundle::from_section(
                                "Aceptar",
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

        let current_turn = game_manager.0.get_turn();
        let current_player = match current_turn {
            Turn::Red => &game_manager.0.get_band_player().red,
            Turn::Black => &game_manager.0.get_band_player().black,
        };

        if *current_player == GamePlayer::Computer {
            let movement = game_manager.0.computer_plays();
            move_piece(
                &movement,
                &mut transform_square_piece,
                &asset_server,
                &mut game_manager,
                &mut piece_by_id,
                &mut commands,
            );
        }
    }
}

pub fn button_system(
    mut interaction_query: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<AcceptButton>)>,
    mut exit: EventWriter<AppExit>,
) {
    for (interaction, mut color) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                exit.send(AppExit);
            }
            Interaction::Hovered => {
                *color = Color::rgb(0.6, 0.6, 0.6).into();
            }
            Interaction::None => {
                *color = Color::rgb(0.5, 0.5, 0.5).into();
            }
        }
    }
}
