from flask_book_recommendation.app import create_app
app = create_app()
for rule in app.url_map.iter_rules():
    print(f"{rule.endpoint}: {rule.rule} [{', '.join(rule.methods)}]")
