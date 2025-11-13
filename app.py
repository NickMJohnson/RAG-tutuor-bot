from shiny import App, reactive, render, req, ui

app_ui = ui.page_fillable(
  ui.input_slider("n", "N", 0, 100, 20),
  ui.output_text_verbatim("txt"),
)


def server(input, output, session):
  @render.text
  def txt():
    return f"n*2 is {input.n() * 2}"


app = App(app_ui, server)
