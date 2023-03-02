import {
  Outlet,
  RouterProvider,
  createReactRouter,
  createRouteConfig,
} from "@tanstack/react-router";

import StringMatch from "./views/StringMatch";

import "./App.css";
import Home from "./views/Home";
import WindowMerge from "./views/WindowMerge";
import Embedding from "./views/Embedding";
import Neural from "./views/Neural";

const routeConfig = createRouteConfig().createChildren((createRoute) => [
  createRoute({
    path: "/",
    component: Home,
  }),
  createRoute({
    path: "/string-match",
    component: StringMatch,
  }),
  createRoute({
    path: "/window-merge",
    component: WindowMerge,
  }),
  createRoute({
    path: "/embedding",
    component: Embedding,
  }),
  createRoute({
    path: "/neural",
    component: Neural,
  }),
]);

const router = createReactRouter({ routeConfig });

function App() {
  return (
    <RouterProvider router={router}>
      <div className="App">
        <Outlet />
      </div>
    </RouterProvider>
  );
}

export default App;
