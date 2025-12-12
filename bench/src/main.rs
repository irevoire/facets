use index::{Filter, Index, SearchQuery};

static DATA: &str = include_str!("players.csv");

fn main() {
    let lines = DATA;
    let now = std::time::Instant::now();
    let index = Index::from_csv(lines.to_string(), b',').unwrap();
    println!("Indexed in {:?}", now.elapsed());

    let mut query = SearchQuery::new();
    query.with_filter(Filter::less_than("Age".to_string(), 19.into()));

    println!("Query: {query}");
    let now = std::time::Instant::now();
    let results = index.execute(&query).unwrap();
    let time = now.elapsed();
    println!("Found {} results in {time:?}", results.len());
}
