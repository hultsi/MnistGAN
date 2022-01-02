#pragma once

namespace variadics {
    template <int T, int... Args>
    struct Sum { static constexpr int value = T + Sum<Args...>::value; };
    template <int T>
    struct Sum <T> { static constexpr int value = T; };
    
    template <int T, int... Args>
    struct Last { static constexpr int value = Last<Args...>::value; };
    template <int T>
    struct Last <T> { static constexpr int value = T; };
}

// // Helpers
// template<typename, typename>
// struct concat { };

// template<typename... Ts, typename... Us>
// struct concat<std::tuple<Ts...>, std::tuple<Us...>>
// {
//     using type = std::tuple<Ts..., Us...>;
// };


// // Remove last tuple element
// template <class T>
// struct last;

// template <class T>
// struct last<std::tuple<T>>
// {
//     using type = std::tuple<>;
// };

// template <typename T, typename... ARGS>
// struct last<std::tuple<T, ARGS...>>
// {
//     using type = typename concat<
//         std::tuple<T>,
//         typename last<std::tuple<ARGS...>>::type
//     >::type;
// };