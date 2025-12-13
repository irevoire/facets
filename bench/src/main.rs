use std::time::Duration;

use facets::key::Key;
use index::{Filter, Index, IndexConfig, SearchQuery, StorageKind};

use clap::{Parser, ValueEnum};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum StorageKindArg {
    Simple,
    Saute,
    BTree,
}

/// Benchmark tests
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Type of storage to use
    #[arg(short, long, value_enum)]
    storage: Option<StorageKindArg>,

    /// Size of BTree (only)
    #[arg(short, long)]
    btree_order: Option<usize>,

    /// Number of lines to test on
    #[arg(short, long)]
    lines: Option<usize>,
}

static DATA: &str = include_str!("players.csv");

fn get_data(count: Option<usize>) -> String {
    match count {
        Some(count) => DATA.lines().take(count).collect::<Vec<_>>().join("\n"),
        None => DATA.to_string(),
    }
}

fn get_configs() -> Vec<IndexConfig> {
    let mut configurations = vec![];

    configurations.push(IndexConfig {
        storage: index::StorageKind::Simple,
        ..Default::default()
    });

    let mut order = 1;
    while order <= 1024 {
        configurations.push(IndexConfig {
            storage: index::StorageKind::BTree(order),
            ..Default::default()
        });
        order *= 2;
    }

    configurations
}

fn main() {
    let args = Args::parse();
    let lines = get_data(args.lines);

    let reference = Index::from_csv(lines.to_string(), b',', None, None).unwrap();

    let check_fields = reference
        .fields()
        .iter()
        // .skip(10)
        // .take(30)
        .cloned()
        .collect::<Vec<_>>();

    enum QueryKind {
        Range,
        LessThan,
        LessThanOrEqual,
        GreaterThan,
        GreaterThanOrEqual,
        Equal,
        Not,
    }
    let mut kind = QueryKind::LessThan;
    let mut string_equal = true;
    let queries = check_fields
        .iter()
        .map(|field| {
            let entry = reference.consult(field, reference.len() / 4);
            let other_entry = reference.consult(field, reference.len() / 2);
            match entry.parse::<usize>() {
                Ok(number) => {
                    let key = Key::from(number);
                    let mut query = SearchQuery::new();
                    match kind {
                        QueryKind::Range => {
                            kind = QueryKind::LessThan;
                            let other_number = other_entry.parse::<usize>().unwrap();
                            let other_key = Key::from(other_number);
                            let (key, other_key) = if number < other_number {
                                (key, other_key)
                            } else {
                                (other_key, key)
                            };
                            query.with_filter(Filter::in_range(field.to_string(), key..other_key));
                        }
                        QueryKind::LessThan => {
                            kind = QueryKind::LessThanOrEqual;
                            query.with_filter(Filter::less_than(field.to_string(), key))
                        }
                        QueryKind::LessThanOrEqual => {
                            kind = QueryKind::GreaterThan;
                            query.with_filter(Filter::less_than(field.to_string(), key))
                        }
                        QueryKind::GreaterThan => {
                            kind = QueryKind::GreaterThanOrEqual;
                            query.with_filter(Filter::greater_than(field.to_string(), key))
                        }
                        QueryKind::GreaterThanOrEqual => {
                            kind = QueryKind::Equal;
                            query.with_filter(Filter::less_than(field.to_string(), key))
                        }
                        QueryKind::Equal => {
                            kind = QueryKind::Not;
                            query.with_filter(Filter::equal(field.to_string(), key))
                        }
                        QueryKind::Not => {
                            kind = QueryKind::Range;
                            query.with_filter(Filter::Not(Box::new(Filter::equal(
                                field.to_string(),
                                key,
                            ))))
                        }
                    }
                    query
                }
                Err(_) => {
                    let key = Key::from(entry);
                    let mut query = SearchQuery::new();
                    if string_equal {
                        query.with_filter(Filter::equal(field.to_string(), key));
                    } else {
                        query.with_filter(Filter::Not(Box::new(Filter::equal(
                            field.to_string(),
                            key,
                        ))))
                    }
                    string_equal = !string_equal;
                    query
                }
            }
        })
        .collect::<Vec<_>>();

    let results = queries
        .iter()
        .clone()
        .map(|query| reference.execute(&query).unwrap().len())
        .collect::<Vec<_>>();

    let configurations = match args.storage {
        Some(storage) => match storage {
            StorageKindArg::Simple => vec![IndexConfig {
                storage: StorageKind::Simple,
                ..Default::default()
            }],
            StorageKindArg::Saute => vec![IndexConfig {
                storage: StorageKind::Saute,
                ..Default::default()
            }],
            StorageKindArg::BTree => match args.btree_order {
                Some(order) => vec![IndexConfig {
                    storage: StorageKind::BTree(order),
                    ..Default::default()
                }],
                None => vec![IndexConfig::default()],
            },
        },
        None => get_configs(),
    };

    for config in configurations {
        println!("Running with config: {config:?}");
        let now = std::time::Instant::now();
        let index = Index::from_csv(lines.to_string(), b',', None, Some(config)).unwrap();
        println!("Indexed in {:?}", now.elapsed());
        let now = std::time::Instant::now();
        while now.elapsed() < Duration::from_secs(10) {
            for (i, query) in queries.iter().enumerate().clone() {
                assert_eq!(index.execute(&query).unwrap().len(), results[i]);
            }
        }
        let time = now.elapsed();
        println!("Queried in {time:?}\n");
        // println!(
        //     "Here are {} results: {}",
        //     if results.len() < 20 {
        //         "the"
        //     } else {
        //         "the first 20"
        //     },
        //     results
        //         .take(20)
        //         .map(|result| result.1[2].1.clone())
        //         .collect::<Vec<_>>()
        //         .join(", ")
        // );
    }
}
