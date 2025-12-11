use facets::query::{self, Query};

use index::{Index, SearchQuery};

static DATA: &str = include_str!("players.csv");

fn main() {
    let lines = DATA;
    let index = Index::from_csv(lines.to_string(), b',').unwrap();

    let mut query = SearchQuery::new();
    query.query_on("Age".to_string(), Query::less_than(19.into()));
    println!("Query: {query}");
    let now = std::time::Instant::now();
    let results = index.query(query);
    let time = now.elapsed();
    println!("Found {} results in {time:?}", results.len());
}
